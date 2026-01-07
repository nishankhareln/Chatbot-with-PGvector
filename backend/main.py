from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import os
from typing import Optional, List
from io import BytesIO
import traceback

from database import db
from document_service import document_service  # ← FIXED: removed 's'
from rag_service import rag_service

# Create FastAPI app
app = FastAPI(
    title="RAG Assistant API",
    description="Document Q&A System with RAG Architecture",
    version="1.0.0"
)

# CORS middleware - FIXED: proper configuration for development
app.add_middleware(  # ← FIXED: no line break
    CORSMiddleware,
    allow_origins=["*"],  # ← FIXED: allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # ← FIXED: allow all methods
    allow_headers=["*"],  # ← FIXED: allow all headers
)

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    document_id: Optional[int] = None
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: list

class UploadResponse(BaseModel):
    message: str
    documents: List[dict]
    total_documents: int
    total_chunks: int

# ========================================
# FRONTEND ENDPOINT (NEW)
# ========================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the HTML frontend"""
    try:
        html_path = os.path.join("templates", "app.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content="""
                <html>
                    <body>
                        <h1>RAG Assistant API</h1>
                        <p>Frontend file not found. Please create templates/app.html</p>
                        <p><a href="/docs">View API Documentation</a></p>
                    </body>
                </html>
                """,
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(content=f"Error loading frontend: {str(e)}", status_code=500)

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    db_healthy = db.health_check()
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": "connected" if db_healthy else "disconnected"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_documents = []
    total_chunks = 0
    
    for uploaded_file in files:
        try:
            file_extension = uploaded_file.filename.split('.')[-1].lower()
            allowed_extensions = ['pdf', 'txt', 'md', 'markdown']
            
            if file_extension not in allowed_extensions:
                processed_documents.append({
                    "filename": uploaded_file.filename,
                    "status": "failed",
                    "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                })
                continue
            
            file_content = await uploaded_file.read()
            
            if len(file_content) == 0:
                processed_documents.append({
                    "filename": uploaded_file.filename,
                    "status": "failed",
                    "error": "Empty file"
                })
                continue
            
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            
            print(f"Processing file: {uploaded_file.filename}")
            chunks = document_service.process_document(file_path, file_extension)
            
            document_id = db.insert_document(
                uploaded_file.filename, 
                file_extension,
                file_content
            )
            
            rag_service.embed_and_store_chunks(document_id, chunks)
            os.remove(file_path)
            
            processed_documents.append({
                "filename": uploaded_file.filename,
                "document_id": document_id,
                "chunks_count": len(chunks),
                "file_size": len(file_content),
                "status": "success"
            })
            
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"Error processing {uploaded_file.filename}: {e}")
            processed_documents.append({
                "filename": uploaded_file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    successful_docs = [doc for doc in processed_documents if doc.get("status") == "success"]
    
    if not successful_docs:
        raise HTTPException(
            status_code=400, 
            detail="No documents were successfully processed"
        )
    
    return UploadResponse(
        message=f"Successfully processed {len(successful_docs)} out of {len(files)} documents",
        documents=processed_documents,
        total_documents=len(successful_docs),
        total_chunks=total_chunks
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about uploaded documents"""
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        document_id = request.document_id
        
        if document_id is None:
            latest_id = db.get_latest_document_id()
            if latest_id is None:
                raise HTTPException(
                    status_code=404, 
                    detail="No documents found. Please upload a document first."
                )
        
        result = rag_service.query(
            question=request.question,
            document_id=document_id,
            top_k=request.top_k
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/documents")
def list_documents():
    """Get list of all uploaded documents"""
    try:
        documents = db.get_all_documents()
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}")
def get_document_info(document_id: int):
    """Get information about a specific document"""
    try:
        doc_info = db.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/download")
def download_document(document_id: int):
    """Download the original document file"""
    try:
        doc_data = db.get_document_file(document_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_stream = BytesIO(bytes(doc_data['file_data']))
        
        media_types = {
            'pdf': 'application/pdf',
            'txt': 'text/plain',
            'md': 'text/markdown',
            'markdown': 'text/markdown'
        }
        media_type = media_types.get(doc_data['file_type'], 'application/octet-stream')
        
        return StreamingResponse(
            file_stream,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={doc_data['filename']}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}")
def delete_document(document_id: int):
    """Delete a document and all its chunks"""
    try:
        db.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)