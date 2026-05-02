"""
FastAPI entry point.
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from database import db
from document_service import document_service, CODE_EXTENSIONS, TEXT_EXTENSIONS
from rag_service import rag_service

app = FastAPI(
    title="Advanced RAG Assistant",
    description="Hybrid search + reranker + contextual retrieval + AST chunking",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"} | TEXT_EXTENSIONS | CODE_EXTENSIONS


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    document_id: Optional[int] = None
    top_k: int = 5
    session_id: Optional[str] = None
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list
    diagnostics: dict


class UploadResponse(BaseModel):
    message: str
    documents: List[dict]
    total_documents: int
    total_chunks: int


# -------------------------- frontend --------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join("templates", "app.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        "<h1>Advanced RAG API</h1><p><a href='/docs'>Swagger</a></p>",
        status_code=200,
    )


# -------------------------- health --------------------------
@app.get("/health")
def health_check():
    db_ok = db.health_check()
    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database": "connected" if db_ok else "disconnected",
        "reranker": rag_service.reranker is not None,
        "contextual_retrieval": bool(os.getenv("USE_CONTEXTUAL_RETRIEVAL", "1") == "1"),
    }


# -------------------------- upload --------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    processed: List[dict] = []
    total_chunks = 0

    for uploaded in files:
        try:
            ext = uploaded.filename.rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                processed.append({
                    "filename": uploaded.filename,
                    "status": "failed",
                    "error": f"Unsupported type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                })
                continue

            content = await uploaded.read()
            if not content:
                processed.append({
                    "filename": uploaded.filename,
                    "status": "failed",
                    "error": "Empty file",
                })
                continue

            tmp_path = os.path.join(UPLOAD_DIR, uploaded.filename)
            with open(tmp_path, "wb") as f:
                f.write(content)

            try:
                chunks = document_service.process_document(tmp_path, ext)
                doc_id = db.insert_document(uploaded.filename, ext, content)
                rag_service.embed_and_store_chunks(doc_id, chunks)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            processed.append({
                "filename": uploaded.filename,
                "document_id": doc_id,
                "chunks_count": len(chunks),
                "file_size": len(content),
                "status": "success",
            })
            total_chunks += len(chunks)

        except Exception as e:
            print(f"[upload] {uploaded.filename}: {e}")
            processed.append({
                "filename": uploaded.filename,
                "status": "failed",
                "error": str(e),
            })

    successes = [d for d in processed if d["status"] == "success"]
    if not successes:
        raise HTTPException(status_code=400, detail="No documents were processed successfully")

    return UploadResponse(
        message=f"Processed {len(successes)} of {len(files)} files",
        documents=processed,
        total_documents=len(successes),
        total_chunks=total_chunks,
    )


# -------------------------- chat --------------------------
def _resolve_document_id(req_doc_id: Optional[int]) -> Optional[int]:
    if req_doc_id is not None:
        return req_doc_id
    latest = db.get_latest_document_id()
    if latest is None:
        raise HTTPException(status_code=404, detail="No documents — upload one first.")
    # None = search across all docs by default; we explicitly return None to enable that.
    return None


def _load_history(req: ChatRequest) -> list:
    if req.session_id:
        return [dict(role=m["role"], content=m["content"]) for m in db.get_history(req.session_id)]
    return [m.model_dump() for m in req.history]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    doc_id = _resolve_document_id(req.document_id)
    history = _load_history(req)

    if req.session_id:
        db.append_message(req.session_id, "user", req.question)

    result = rag_service.query(
        question=req.question,
        document_id=doc_id,
        top_k=req.top_k,
        history=history,
    )

    if req.session_id:
        db.append_message(req.session_id, "assistant", result["answer"])

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        diagnostics=result["diagnostics"],
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    doc_id = _resolve_document_id(req.document_id)
    history = _load_history(req)

    if req.session_id:
        db.append_message(req.session_id, "user", req.question)

    stream, sources, diagnostics = rag_service.query_stream(
        question=req.question,
        document_id=doc_id,
        top_k=req.top_k,
        history=history,
    )

    def event_gen():
        meta = {"type": "meta", "sources": sources, "diagnostics": diagnostics}
        yield f"data: {json.dumps(meta)}\n\n"
        full_answer_parts: List[str] = []
        for token in stream:
            full_answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        full_answer = "".join(full_answer_parts)
        if req.session_id:
            try:
                db.append_message(req.session_id, "assistant", full_answer)
            except Exception as e:
                print(f"[chat/stream] history save failed: {e}")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# -------------------------- documents --------------------------
@app.get("/documents")
def list_documents():
    docs = db.get_all_documents()
    return {"documents": docs, "count": len(docs)}


@app.get("/document/{document_id}")
def get_document_info(document_id: int):
    info = db.get_document_info(document_id)
    if not info:
        raise HTTPException(status_code=404, detail="Document not found")
    return info


@app.get("/document/{document_id}/download")
def download_document(document_id: int):
    doc = db.get_document_file(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    media_types = {
        "pdf": "application/pdf",
        "txt": "text/plain",
        "md": "text/markdown",
        "markdown": "text/markdown",
    }
    media_type = media_types.get(doc["file_type"], "application/octet-stream")
    return StreamingResponse(
        BytesIO(bytes(doc["file_data"])),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{doc["filename"]}"'},
    )


@app.delete("/document/{document_id}")
def delete_document(document_id: int):
    db.delete_document(document_id)
    return {"message": f"Document {document_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
