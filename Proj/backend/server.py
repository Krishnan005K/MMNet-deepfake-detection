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
import csv
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
# Utility: PDF generation
# -----------------------------



def generate_pdf_report(video_name, csv_path, output_pdf_path, results):

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
    # -----------------------------
    # FRAME LEVEL ANALYSIS
    # -----------------------------
    elements.append(Paragraph("<b>===== Frame-Level Detection =====</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    fake_count = 0

    if not os.path.exists(csv_path):
        elements.append(Paragraph("No fake frames detected.", normal_style))
    else:
        with open(csv_path, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            fake_count = len(rows)

            elements.append(Paragraph(f"<b>Total Suspicious Frames:</b> {fake_count}", normal_style))
            elements.append(Spacer(1, 0.3 * inch))

            for row in rows:
                frame_no = row["frame"]
                time_sec = row["time_sec"]
                score = row["score"]
                image_path = os.path.normpath(row["image_path"])

                frame_info = f"""
                <b>Frame:</b> {frame_no}<br/>
                <b>Timestamp:</b> {time_sec} sec<br/>
                <b>Fake Score:</b> {score}
                """

                elements.append(Paragraph(frame_info, normal_style))
                elements.append(Spacer(1, 0.2 * inch))

                if os.path.exists(image_path):
                    img = Image(image_path, width=1 * inch, height=0.75 * inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2 * inch))
                else:
                    elements.append(Paragraph("Image not found.", normal_style))
                    elements.append(Spacer(1, 0.3 * inch))

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

    pdf_path = os.path.join(
        BASE_DIR,
        "reports",
        base_name,
        f"{base_name}_report.pdf"
    )

    generate_pdf_report(
    video_name=base_name,
    csv_path=csv_path,
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
                if file.endswith((".txt", ".csv", ".pdf")):
                    all_reports.append({
                        "filename": file,
                        "folder": folder
                    })
    return {"reports": all_reports}