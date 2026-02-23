from fastapi import FastAPI, UploadFile, File
import os
import shutil
import subprocess
import sys
from io import StringIO
from datetime import datetime

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

    video_path = os.path.join(UPLOAD_DIR, video_filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    results = detect(video_path)

    sys.stdout = old_stdout
    terminal_output = mystdout.getvalue()

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{base_name}_{timestamp}.txt"
    report_path = os.path.join(REPORT_DIR, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(terminal_output)

    return {
        "video": video_filename,
        "report_file": report_filename,
        "raw_report": terminal_output,
        "frames": results.get("frame_level_points", [])
    }
# -----------------------------
# Reports History
# -----------------------------
@app.get("/reports")
def get_reports():

    files = sorted(
        os.listdir(REPORT_DIR),
        reverse=True  # latest first
    )

    reports = []

    for file in files:
        if file.endswith(".txt"):
            reports.append({
                "filename": file
            })

    return {"reports": reports}