import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import process_and_store_pdf, ask_question
from utils.converter import jpeg_to_pdf, csv_to_pdf

load_dotenv()

app = FastAPI(title="AI Support Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "AI Support Chatbot is running"}


@app.post("/ask")
def ask(q: Question):
    try:
        result = ask_question(q.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks_stored = process_and_store_pdf(filepath)
    return {"message": f"Processed '{file.filename}'", "chunks_stored": chunks_stored}


@app.post("/convert/jpeg-to-pdf")
async def convert_jpeg(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPEG files allowed")
    jpeg_path = os.path.join(UPLOAD_DIR, file.filename)
    pdf_path = jpeg_path.rsplit(".", 1)[0] + ".pdf"
    with open(jpeg_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jpeg_to_pdf(jpeg_path, pdf_path)
    chunks_stored = process_and_store_pdf(pdf_path)
    return {"message": f"Converted and indexed '{file.filename}'", "pdf": os.path.basename(pdf_path), "chunks_stored": chunks_stored}


@app.post("/convert/csv-to-pdf")
async def convert_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    csv_path = os.path.join(UPLOAD_DIR, file.filename)
    pdf_path = csv_path.replace(".csv", ".pdf")
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    csv_to_pdf(csv_path, pdf_path)
    chunks_stored = process_and_store_pdf(pdf_path)
    return {"message": f"Converted and indexed '{file.filename}'", "pdf": os.path.basename(pdf_path), "chunks_stored": chunks_stored}