from fastapi import FastAPI, UploadFile, File
import os, shutil, subprocess
from deepfake_full_pipeline import detect
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, video.filename)

    # save uploaded video
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # convert to WAV
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    subprocess.call(f"ffmpeg -y -i \"{video_path}\" -vn -acodec pcm_s16le -ar 22050 -ac 1 \"{audio_path}\"", shell=True)

    # Capture printed output from detect()
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    results = detect(video_path, audio_path)

    sys.stdout = old_stdout
    terminal_output = mystdout.getvalue()

    return {
        "video": video.filename,
        "audio_used": os.path.basename(audio_path),
        "raw_report": terminal_output  # send terminal text to frontend
    }
