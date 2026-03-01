from fastapi import FastAPI, UploadFile, File
import os
import shutil
import sys
from io import StringIO
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import librosa
import csv
import moviepy.editor as mp_editor
import numpy as np
import cv2
import mediapipe as mp
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import ListFlowable, ListItem
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from deepfake_full_pipeline import detect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Directories
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# Static
# -----------------------------
app.mount("/frames", StaticFiles(directory=FRAME_DIR), name="frames")
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")


# ---------------------------------------
# 1️⃣ Extract Audio
# ---------------------------------------
def extract_audio(video_path, output_audio_path):
    clip = mp_editor.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, codec="pcm_s16le")
    return output_audio_path


# ---------------------------------------
# 2️⃣ MFCC Heatmap
# ---------------------------------------
def generate_mfcc_heatmap(audio_path, output_dir):
    y, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    heatmap_path = os.path.join(output_dir, "mfcc_heatmap.png")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis="time", sr=sr)
    plt.colorbar()
    plt.title("MFCC Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return heatmap_path


# ---------------------------------------
# 3️⃣ LipSync + Audio Energy Graph
# ---------------------------------------
def generate_lipsync_graph(video_path, audio_path, output_dir):

    # --- Audio Energy ---
    y, sr = librosa.load(audio_path)
    frame_hop = int(sr / 30)
    energy = librosa.feature.rms(y=y, hop_length=frame_hop)[0]
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)

    # --- Lip Distance ---
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    lip_distances = []

    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                upper = lm.landmark[13]
                lower = lm.landmark[14]

                dist = np.linalg.norm(
                    np.array([upper.x * w, upper.y * h]) -
                    np.array([lower.x * w, lower.y * h])
                )
                lip_distances.append(dist)
            else:
                lip_distances.append(0)

    cap.release()

    lip_distances = np.array(lip_distances)
    lip_distances = (lip_distances - np.min(lip_distances)) / \
                    (np.max(lip_distances) - np.min(lip_distances) + 1e-8)

    min_len = min(len(lip_distances), len(energy))
    lip_distances = lip_distances[:min_len]
    energy = energy[:min_len]

    graph_path = os.path.join(output_dir, "lipsync_energy_graph.png")

    plt.figure(figsize=(10, 4))
    plt.plot(lip_distances, label="Lip Distance")
    plt.plot(energy, label="Audio Energy")
    plt.legend()
    plt.title("Audio vs Lip Movement Comparison")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return graph_path

# -----------------------------
# Utility: PDF generation
# -----------------------------



def generate_pdf_report(video_name, csv_path, output_pdf_path, results, heatmap_path,lipsync_graph_path):

    output_dir = os.path.dirname(output_pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    normal_style = styles["Normal"]

    # -----------------------------
    # TITLE
    # -----------------------------
    elements.append(Paragraph("Deepfake Detection Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Video Name:</b> {video_name}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

   # -----------------------------
    # SUMMARY SECTION
    # -----------------------------
    elements.append(Paragraph("<b>===== Analysis Summary =====</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    video_path = results.get("video")
    model_path = results.get("model")
    audio_score = results.get("audio_fake_p")
    lip_sync_score = results.get("lip_sync_fake_p")
    video_art_score = results.get("video_artifact_fake_p")
    final_fake = results.get("final_fake_p")
    authenticity = results.get("authenticity")
    verdict = results.get("verdict")
    num_frames = results.get("num_suspicious_frames")
  #  print("DEBUG: Results in PDF Generation ->", results)
    summary_text = f"""
    <b>Video Path:</b> {video_path}<br/>
    <b>Model Used:</b> {model_path}<br/><br/>

    <b>Audio Fake Probability:</b> {audio_score:.4f}<br/>
    <b>Lip Sync Manipulation Probability:</b> {lip_sync_score:.4f}<br/>
    <b>Video Artifact (MMNet) Probability:</b> {video_art_score:.4f}<br/><br/>

    <b>Final Fake Probability:</b> {final_fake:.4f}<br/>
    <b>Authenticity Score:</b> {authenticity:.4f}<br/>
    <b>Total Suspicious Frames:</b> {num_frames}<br/><br/>

    <b>Final Verdict:</b> {verdict}
    """

    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 0.4 * inch))
     # ADD MFCC GRAPH
    elements.append(Paragraph("MFCC Heatmap Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(heatmap_path, width=5 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.4 * inch))

    # ADD LIP SYNC GRAPH
    elements.append(Paragraph("Lip Sync vs Audio Energy", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(lipsync_graph_path, width=5 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.4 * inch))


    # ---------------------------------------------
    # FRAME LEVEL SECTION (2 IMAGES PER ROW)
    # ---------------------------------------------
    elements.append(Paragraph("<b>===== Frame-Level Detection =====</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    image_row = []
    table_data = []

    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding="utf-8") as csvfile:
            reader = list(csv.DictReader(csvfile))

            for idx, row in enumerate(reader):

                frame_info = f"""
                <b>Frame:</b> {row['frame']}<br/>
                <b>Timestamp:</b> {row['time_sec']} sec<br/>
                <b>Fake Score:</b> {row['score']}
                """

                if os.path.exists(row["image_path"]):

                    img = Image(row["image_path"], width=2.5*inch, height=1.8*inch)

                    # Combine text + image vertically
                    cell = [
                        Paragraph(frame_info, styles["Normal"]),
                        Spacer(1, 0.1 * inch),
                        img
                    ]

                    image_row.append(cell)

                    # When 2 images collected → make one row
                    if len(image_row) == 2:
                        table_data.append(image_row)
                        image_row = []

            # If odd number → last single image row
            if len(image_row) == 1:
                table_data.append(image_row)

        if table_data:
            table = Table(table_data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
                ('INNERGRID', (0,0), (-1,-1), 0.5, colors.grey),
            ]))

            elements.append(table)
        # Set metadata
    doc.title = f"Deepfake Report - {video_name}"
    doc.author = "Deepfake Detection System"
    doc.subject = "AI Deepfake Analysis Report"
    doc.creator = "Deepfake Backend Service"

    doc.build(elements)

    
# -----------------------------
# Root
# -----------------------------
@app.get("/")
def root():
    return {"message": "Deepfake Detection API running"}

# -----------------------------
# Analyze
# -----------------------------
@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    video_filename = video.filename
    base_name = os.path.splitext(video_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_path = os.path.join(UPLOAD_DIR, video_filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    results = detect(video_path)
    sys.stdout = old_stdout
    terminal_output = mystdout.getvalue()

    # ---- Create report folder ----
    video_report_folder = os.path.join(REPORT_DIR, base_name)
    os.makedirs(video_report_folder, exist_ok=True)

    # TXT report
    txt_filename = f"{base_name}_{timestamp}.txt"
    txt_path = os.path.join(video_report_folder, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(terminal_output)

    # CSV report
    csv_filename = f"{base_name}_deepfake_report.csv"
    csv_path = os.path.join(video_report_folder, csv_filename)
    results['csv_path'] = csv_path
    if 'frame_level_points' in results:
        df = pd.DataFrame(results['frame_level_points'])
        df.to_csv(csv_path, index=False)

    # PDF report
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    report_folder = os.path.join(REPORT_DIR, base_name)
    os.makedirs(report_folder, exist_ok=True)
    pdf_path = os.path.join(
        BASE_DIR,
        "reports",
        base_name,
        f"{base_name}_report.pdf"
    )
    audio_path = os.path.join(UPLOAD_DIR, f"{base_name}.wav")
    heatmap_path = generate_mfcc_heatmap(audio_path, report_folder)
    lipsync_graph_path = generate_lipsync_graph(video_path, audio_path, report_folder)

    
    generate_pdf_report(
    video_name=base_name,
    csv_path=csv_path,
    heatmap_path=heatmap_path,
    lipsync_graph_path=lipsync_graph_path,
    output_pdf_path=pdf_path,
    results=results
    )

    return {
        "video": video_filename,
        "txt_file": f"reports/{base_name}/{txt_filename}",
        "csv_file": f"reports/{base_name}/{csv_filename}",
        "pdf_file": f"reports/{base_name}/{os.path.basename(pdf_path)}",
        "raw_report": terminal_output,
        "frames": results.get("frame_level_points", [])
    }

# -----------------------------
# Reports History
# -----------------------------
@app.get("/reports")
def get_reports():
    folders = sorted(os.listdir(REPORT_DIR), reverse=True)
    all_reports = []
    for folder in folders:
        folder_path = os.path.join(REPORT_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith((".pdf")):
                    all_reports.append({
                        "filename": file,
                        "folder": folder
                    })
    return {"reports": all_reports}