import os
import json
import shutil
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
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

# ── Auth ─────────────────────────────────────────────
API_KEY = os.getenv("APP_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ── Models ───────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class Question(BaseModel):
    query: str
    history: Optional[List[Message]] = []


# ── Routes ───────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AI Support Chatbot is running"}


@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.post("/ask")
def ask(q: Question, key: str = Depends(verify_api_key)):
    try:
        result = ask_question(q.query, q.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_stream(q: Question, key: str = Depends(verify_api_key)):
    from rag import voyage, index, claude

    query_embedding = voyage.embed([q.query], model="voyage-3", input_type="query").embeddings[0]
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n".join(context_chunks)

    messages = [{"role": m.role, "content": m.content} for m in q.history]
    messages.append({
        "role": "user",
        "content": f"""Answer based only on this context:
{context}

Question: {q.query}"""
    })

    def generate():
        with claude.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="You are a helpful support assistant.",
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...), key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks_stored = process_and_store_pdf(filepath)
    return {"message": f"Processed '{file.filename}'", "chunks_stored": chunks_stored}


@app.post("/convert/jpeg-to-pdf")
async def convert_jpeg(file: UploadFile = File(...), key: str = Depends(verify_api_key)):
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
async def convert_csv(file: UploadFile = File(...), key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    csv_path = os.path.join(UPLOAD_DIR, file.filename)
    pdf_path = csv_path.replace(".csv", ".pdf")
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    csv_to_pdf(csv_path, pdf_path)
    chunks_stored = process_and_store_pdf(pdf_path)
    return {"message": f"Converted and indexed '{file.filename}'", "pdf": os.path.basename(pdf_path), "chunks_stored": chunks_stored}