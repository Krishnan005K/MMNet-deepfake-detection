#!/usr/bin/env python3
"""
deepfake_full_pipeline.py

Usage:
    python deepfake_full_pipeline.py --video videoplayback.mp4 --model 4589fd69-d042-4a53-8067-eba66670d614.pth

Produces:
 - printed summary
 - CSV file: <video_basename>_deepfake_report.csv
"""

import os
import argparse
import csv
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import librosa

import cv2
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

# from moviepy.editor import VideoFileClip

# Try optional imports
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False

# -------------------------
# 1) Audio model (your model)
# -------------------------
class FakeAudioDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # expects input shape (B, 40, T)
        x = x.unsqueeze(1)   # (B,1,40,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)






def generate_pdf_report(video_name, csv_path, frames_dir, report_dir):
    """
    video_name: original video filename
    csv_path: path to CSV report
    frames_dir: path to video frames folder
    report_dir: where PDF should be saved
    """

    # ------------------------
    # Load CSV
    # ------------------------
    df = pd.read_csv(csv_path)
    
    total_frames = len(df)
    fake_frames = df[df['final_fake_p'] > 0.5]  # threshold for fake
    fake_count = len(fake_frames)
    fake_percentage = (fake_count / total_frames) * 100 if total_frames > 0 else 0
    verdict = df['verdict'][0] if 'verdict' in df.columns else "UNKNOWN"
    
    # ------------------------
    # PDF Setup
    # ------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(video_name)[0]
    pdf_filename = f"{base_name}_{timestamp}_report.pdf"
    pdf_path = os.path.join(report_dir, pdf_filename)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ------------------------
    # Cover Page
    # ------------------------
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "🎭 Deepfake Detection Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Video: {video_name}", ln=True)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Verdict: {verdict}", ln=True)
    pdf.cell(0, 10, f"Total Frames: {total_frames}", ln=True)
    pdf.cell(0, 10, f"Fake Frames Detected: {fake_count} ({fake_percentage:.2f}%)", ln=True)
    pdf.ln(5)
    
    # ------------------------
    # Plot Fake Probability Graph
    # ------------------------
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['final_fake_p'], label='Fake Probability')
    plt.xlabel("Frame Number")
    plt.ylabel("Final Fake Probability")
    plt.title("Fake Probability Over Frames")
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(report_dir, f"{base_name}_fake_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    pdf.image(plot_path, w=180)
    pdf.ln(5)
    
    # ------------------------
    # Add Sample Fake Frames
    # ------------------------
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Sample Manipulated Frames:", ln=True)
    
    sample_frames = fake_frames.head(10)  # take first 10 fake frames
    for _, row in sample_frames.iterrows():
        # Frame filename is like: frame_{frameNumber}_{frameSeconds}.jpg
        frame_number = int(row['suspicious_frame_numbers'].strip("[]").split(",")[0])
        frame_sec = float(row['suspicious_timestamps_sec'].strip("[]").split(",")[0])
        frame_file = os.path.join(frames_dir, base_name, f"frame_{frame_number}_{frame_sec:.2f}.jpg")
        if os.path.exists(frame_file):
            pdf.image(frame_file, w=90)
            pdf.ln(5)
    
    # ------------------------
    # XAI-style textual summary
    # ------------------------
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Manipulation Analysis (XAI-style):", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", '', 12)
    
    summary_text = f"""
    The video contains {fake_count} frames with potential manipulations.
    The highest probability of fake frame: {df['final_fake_p'].max():.2f}.
    The lowest probability of fake frame: {df['final_fake_p'].min():.2f}.
    
    Key observations:
    - Lip Sync anomalies: average {df['lip_sync_fake_p'].mean():.2f}
    - Audio manipulation probability: average {df['audio_fake_p'].mean():.2f}
    - Video artifacts detected: average {df['video_artifact_fake_p'].mean():.2f}
    
    Frames with highest manipulation probability have been highlighted above.
    """
    pdf.multi_cell(0, 6, summary_text)
    
    # ------------------------
    # Save PDF
    # ------------------------
    pdf.output(pdf_path)
    
    # Clean up plot image
    if os.path.exists(plot_path):
        os.remove(plot_path)
    
    return pdf_path

def load_audio_model(model_path, device):
    model = FakeAudioDetector().to(device)
    # load state dict (handles both state_dict or full model saved)
    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and not any(k.startswith('_') for k in sd.keys()) and \
       not hasattr(sd, 'state_dict'):
        # assume it's a state_dict
        try:
            model.load_state_dict(sd)
        except Exception as e:
            # try if saved as {'model_state': ...}
            if 'model_state' in sd:
                model.load_state_dict(sd['model_state'])
            else:
                raise
    else:
        # maybe full model saved; attempt load
        try:
            model = sd
        except Exception:
            # fallback: assume state dict loaded already succeeded or failed above
            pass
    model.eval()
    return model

def preprocess_audio_for_model(wav_path, n_mfcc=20, max_len=200):
    """
    Returns a torch tensor shape (1, 40, max_len) float32
    Matching your original preprocessing: n_mfcc=20 + delta => 40 x time
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    # optional: trim leading/trailing silence? left as-is
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([mfcc, delta])  # (40, T)
    # pad/trim to max_len
    if feat.shape[1] < max_len:
        feat = np.pad(feat, ((0,0),(0, max_len - feat.shape[1])), mode='constant')
    else:
        feat = feat[:, :max_len]
    tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 40, max_len)
    return tensor, sr

def predict_audio_fake_prob(model, wav_path, device):
    x, sr = preprocess_audio_for_model(wav_path, n_mfcc=20, max_len=200)
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
    # probs[1] = fake class probability (consistent with your earlier code)
    fake_prob = float(probs[1])
    real_prob = float(probs[0])
    return real_prob, fake_prob

# -------------------------
# 2) Extract audio (MoviePy-based)
# -------------------------
def extract_audio_with_moviepy(video_path, out_wav):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("No audio stream in video.")
    clip.audio.write_audiofile(out_wav, codec='pcm_s16le', verbose=False, logger=None)
    clip.close()
    return out_wav

# -------------------------
# 3) Lip-sync analysis
#    Prefer MediaPipe FaceMesh if available, else OpenCV contour fallback
# -------------------------
def lip_sync_with_mediapipe(video_path, wav_path, debug=False):
    # returns sync_score in [0..1], higher = well-synced
    mp_face_mesh = mp.solutions.face_mesh
    LIP_IDX = [61, 291, 13, 14, 78, 308]  # approximate mesh indices

    def compute_mouth_opening(landmarks, w, h):
        pts = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in LIP_IDX]
        upper = np.mean([pts[2], pts[4]], axis=0)
        lower = np.mean([pts[3], pts[5]], axis=0)
        return max(0.0, lower[1] - upper[1])

    # extract mouth opening time series
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    mouth_series = []
    timestamps = []
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                               refine_landmarks=True, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as fm:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(image)
            t = frame_idx / fps
            if res.multi_face_landmarks:
                mouth_series.append(compute_mouth_opening(res.multi_face_landmarks[0], frame.shape[1], frame.shape[0]))
            else:
                mouth_series.append(0.0)
            timestamps.append(t)
    cap.release()

    # audio envelope (RMS)
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop_length = max(256, int(sr / fps / 2))
    env = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)

    # interpolate audio env to video timestamps
    if len(timestamps) < 3 or len(times) < 3:
        return 0.0
    mouth_arr = np.array(mouth_series)
    env_interp = np.interp(timestamps, times, env)
    # normalize both
    if mouth_arr.std() > 1e-9:
        a = (mouth_arr - mouth_arr.mean()) / mouth_arr.std()
    else:
        a = mouth_arr
    if env_interp.std() > 1e-9:
        b = (env_interp - env_interp.mean()) / env_interp.std()
    else:
        b = env_interp
    # correlation
    if a.std() < 1e-9 or b.std() < 1e-9:
        corr0 = 0.0
    else:
        corr0 = np.corrcoef(a, b)[0,1]
        if np.isnan(corr0):
            corr0 = 0.0
    # final sync score normalized to 0..1
    sync_score = float(np.clip((corr0 + 1) / 2, 0.0, 1.0))
    return sync_score

def lip_sync_opencv_fallback(video_path, wav_path):
    # Simple contour-based mouth-open proxy + audio energy, similar to fallback approach
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    mouth_vals = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        # pick lower center ROI where mouth usually is
        roi = frame[int(h*0.55):int(h*0.85), int(w*0.3):int(w*0.7)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # adaptive threshold to catch dark mouth region
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 9)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0.0
        for c in cnts:
            area = max(area, cv2.contourArea(c))
        mouth_vals.append(area)
    cap.release()

    # get audio envelope
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop_length = max(256, int(sr / fps / 2))
    env = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)
    # map audio frames to video frames by linear interpolation of length
    t_vid = np.linspace(0, len(y)/sr, num=len(mouth_vals))
    env_interp = np.interp(t_vid, times, env[:len(times)]) if len(times)>0 else np.zeros_like(mouth_vals)

    a = np.array(mouth_vals)
    b = env_interp
    if a.std() > 1e-9:
        a = (a - a.mean())/a.std()
    if b.std() > 1e-9:
        b = (b - b.mean())/b.std()
    if a.std() < 1e-9 or b.std() < 1e-9:
        corr0 = 0.0
    else:
        corr0 = np.corrcoef(a, b)[0,1]
        if np.isnan(corr0):
            corr0 = 0.0
    sync_score = float(np.clip((corr0 + 1)/2, 0.0, 1.0))
    return sync_score

def compute_lip_sync_score(video_path, wav_path):
    # try mediapipe first
    try:
        if USE_MEDIAPIPE:
            return lip_sync_with_mediapipe(video_path, wav_path)
        else:
            return lip_sync_opencv_fallback(video_path, wav_path)
    except Exception:
        # fallback robust
        return lip_sync_opencv_fallback(video_path, wav_path)

# -------------------------
# 4) Video artifact heuristic (blur + blink)
# -------------------------
def video_artifact_heuristic(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = 0
    blur_vals = []
    blink_count = 0

    eye_cascade = None
    try:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    except Exception:
        eye_cascade = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_vals.append(lap)

        if eye_cascade is not None:
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 6)
            if len(eyes) == 0:
                blink_count += 1

    cap.release()
    if len(blur_vals) == 0:
        return 0.5
    mean_blur = float(np.mean(blur_vals))
    # normalize  r to [0..1] where lower sharpness => higher fake score
    # clamp extremes heuristically
    blur_score = 1.0 - (min(max(mean_blur, 50.0), 2000.0) - 50.0) / (2000.0 - 50.0)
    # blink score: typical blink fraction ~0.10-0.2 => deviation -> suspicious
    blink_rate = blink_count / max(frames, 1)
    blink_score = min(abs(blink_rate - 0.15) / 0.15, 1.0)
    # combine: higher -> more likely fake
    combined = float(np.clip((blur_score * 0.6 + blink_score * 0.4), 0.0, 1.0))
    return combined

# -------------------------
# Frame-by-frame deepfake analysis (Explainable)
# -------------------------
def frame_level_analysis(video_path, threshold=0.7, sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    frame_id = 0
    suspicious = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(FRAME_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % sample_rate != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 - np.clip((blur - 50) / 1950, 0, 1)

        eyes = eye_cascade.detectMultiScale(gray, 1.1, 6)
        eye_missing = 1.0 if len(eyes) == 0 else 0.0

        fake_score = 0.6 * blur_score + 0.4 * eye_missing

        if fake_score >= threshold:
            time_sec = round(frame_id / fps, 2)

            frame_name = f"frame_{frame_id}_{time_sec}s.jpg"
            frame_path = os.path.join(out_dir, frame_name)

            cv2.imwrite(frame_path, frame)

            suspicious.append({
                "frame": frame_id,
                "time_sec": time_sec,
                "score": round(fake_score, 3),
                "image_path": frame_path
            })

    cap.release()
    return suspicious

# -------------------------
# 5) Overall pipeline
# -------------------------
def run_pipeline(video_path, model_path, out_audio="extracted_audio.wav", device_str=None, debug=False):
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[+] Device: {device}")
    # 1. extract audio
    # print("[+] Extracting audio (MoviePy)...")
    # extract_audio_with_moviepy(video_path, out_audio)

    # 2. load audio model
    print("[+] Loading audio model...")
    model = load_audio_model(model_path, device)

    # 3. audio prediction
    print("[+] Predicting audio deepfake probability...")
    real_p, fake_p = predict_audio_fake_prob(model, out_audio, device)

    # 4. lip-sync
    print("[+] Calculating lip-sync score (may take time)...")
    sync_score = compute_lip_sync_score(video_path, out_audio)   # 0..1 (1 = good sync)
    lip_fake_prob = 1.0 - sync_score
    

    # 5. video artifact
    print("[+] Computing video artifact heuristic...")
    video_fake = video_artifact_heuristic(video_path)
    # Frame-level explainable analysis
    print("[+] Running frame-level explainable analysis...")
    frame_points = frame_level_analysis(video_path)
   


    # 6. combine scores
    # weights: audio 0.6, sync 0.3, video_artifact 0.1
    w_audio, w_sync, w_video = 0.6, 0.3, 0.1
    final_fake_score = float(np.clip(w_audio * fake_p + w_sync * lip_fake_prob + w_video * video_fake, 0.0, 1.0))
    authenticity = 1.0 - final_fake_score

    # 7. print & save CSV
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = base + "_deepfake_report.csv"

    print("\n===== Deepfake Report =====")
    print(f"Video        : {video_path}")
    print(f"Model        : {model_path}")
    print(f"Audio Fake Model : {fake_p:.4f}   (audio model)")
    print(f"Video Fake Model : MMNET model  (video model)")
    print(f"Lip-sync p   : {lip_fake_prob:.4f} (1 - sync_score; 0=good sync)")
    print(f"Video art p  : {video_fake:.4f} (MMNET)")
    print(f"FINAL fake p : {final_fake_score:.4f}  (1 = very likely fake)")
    print(f"Authenticity : {authenticity:.4f}  (1 = likely real)")

    num_frames = len(frame_points)

    if final_fake_score >= 0.75:
        verdict = "HIGHLY LIKELY FAKE"
    elif final_fake_score >= 0.50:
        verdict = "LIKELY FAKE"
    elif final_fake_score >= 0.30:
        verdict = "SUSPICIOUS - POSSIBLE MANIPULATION"
    elif final_fake_score < 0.30 and num_frames > 5:
        verdict = "LOW OVERALL SCORE, BUT MULTIPLE SUSPICIOUS FRAMES DETECTED"
    else:
        verdict = "LIKELY REAL"
    print(f"VERDICT      : {verdict}")

    # -------- Frame-level explainable output --------
  
    print("\n--- Frame-Level Analysis ---")
    print(f"Suspicious Frames Count : {len(frame_points)}")
    print(f"Suspicious Frames : {len(frame_points)}")

    for p in frame_points[:5]:
        print(f" • Time {p['time_sec']}s | Frame {p['frame']} | Score {p['score']}")


    if len(frame_points) == 0:
        print("No significant frame-level manipulation detected.")
    else:
        print("Potential deepfake detected at:")
        for p in frame_points[:10]:  # limit console output
            print(f"  • Time {p['time_sec']}s | Frame {p['frame']} | Score {p['score']}")

        if len(frame_points) > 10:
            print(f"  ...and {len(frame_points) - 10} more suspicious frames")

    print("--------------------------------------")

    # -------- Save CSV --------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "video",
            "model",
            "audio_fake_p",
            "lip_sync_fake_p",
            "video_artifact_fake_p",
            "final_fake_p",
            "authenticity",
            "verdict",
            "num_suspicious_frames",
            "suspicious_timestamps_sec",
            "suspicious_frame_numbers"
        ])

        writer.writerow([
            video_path,
            model_path,
            f"{fake_p:.6f}",
            f"{lip_fake_prob:.6f}",
            f"{video_fake:.6f}",
            f"{final_fake_score:.6f}",
            f"{authenticity:.6f}",
            verdict,
            len(frame_points),
            [p["time_sec"] for p in frame_points],
            [p["frame"] for p in frame_points]
        ])

    print(f"[+] CSV saved: {csv_path}")

    # cleanup audio if temporary
    # os.remove(out_audio)  # keep for debugging
    return {
    "video": video_path,
    "model": model_path,
    "audio_fake_p": fake_p,
    "lip_sync_fake_p": lip_fake_prob,
    "video_artifact_fake_p": video_fake,
    "final_fake_p": final_fake_score,
    "authenticity": authenticity,
    "verdict": verdict,
    "frame_level_points": frame_points,   # 👈 EXPLAINABLE OUTPUT
    "num_suspicious_frames": len(frame_points),
    
}


def detect(video_path, model_path="../fake_audio_detector.pth"):

    base_name = os.path.splitext(video_path)[0]
    audio_path = base_name + ".wav"

    # Extract audio using moviepy (NO system ffmpeg dependency)
    from moviepy.editor import VideoFileClip

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

    return run_pipeline(video_path, model_path, out_audio=audio_path)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full deepfake checker (audio model + lip-sync + mmm).")
    parser.add_argument("--video", required=False, default="videoplayback.mp4", help="Path to video file")
    parser.add_argument("--model", required=False, default="fake_audio_detector.pth", help="Path to audio .pth model")
    parser.add_argument("--out_audio", default="videoplayback.wav", help="Temp audio path")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    run_pipeline(args.video, args.model, out_audio=args.out_audio, device_str=args.device, debug=args.debug)
