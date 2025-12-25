from fastapi import FastAPI, UploadFile, File
import os
import shutil
import subprocess
import sys
from io import StringIO

from deepfake_full_pipeline import detect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
UPLOAD_DIR = "uploads"
FRAME_DIR = "frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# -----------------------------
# Expose frame images to frontend
# -----------------------------
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# -----------------------------
# Root (optional, avoids 404)
# -----------------------------
@app.get("/")
def root():
    return {"message": "Deepfake Detection API running"}

# -----------------------------
# Analyze endpoint
# -----------------------------
@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    # Save uploaded video
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Extract audio
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    subprocess.call(
        f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 22050 -ac 1 "{audio_path}"',
        shell=True
    )

    # Capture printed output from pipeline
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Run deepfake pipeline
    results = detect(video_path, audio_path)

    sys.stdout = old_stdout
    terminal_output = mystdout.getvalue()

    # -----------------------------
    # Extract frame image URLs
    # -----------------------------
    frames = []
    for p in results.get("frame_level_points", []):
        frames.append({
            "frame": p["frame"],
            "time_sec": p["time_sec"],
            "score": p["score"],
            # convert local path -> public URL
            "image_url": f"/{p['image_path'].replace(os.sep, '/')}"
        })

    return {
        "video": video.filename,
        "audio_used": os.path.basename(audio_path),
        "raw_report": terminal_output,   # keeps your existing UI working
        "frames": frames                # NEW: frame images + timestamps
    }
