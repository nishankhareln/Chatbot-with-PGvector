from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from typing import Optional, List
from io import BytesIO

from database import db
from document_services import document_service
from rag_service import rag_service

# Create FastAPI app
app = FastAPI(title="Simple RAG System API")

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists (for temporary processing)
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
    documents: List[dict]  # List of processed documents
    total_documents: int
    total_chunks: int


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Simple RAG System API",
        "endpoints": {
            "health": "/health",
            "upload": "/upload (POST - multiple files)",
            "chat": "/chat (POST)",
            "documents": "/documents (GET)",
            "document_info": "/document/{id} (GET)",
            "download": "/document/{id}/download (GET)",
            "delete": "/document/{id} (DELETE)"
        }
    }


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
    """
    Upload and process multiple documents (PDF, TXT, or Markdown)
    
    Args:
        files: List of uploaded files
        
    Returns:
        Upload status with document_ids for all processed files
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_documents = []
    total_chunks = 0
    
    for uploaded_file in files:
        try:
            # Validate file type
            file_extension = uploaded_file.filename.split('.')[-1].lower()
            allowed_extensions = ['pdf', 'txt', 'md', 'markdown']
            
            if file_extension not in allowed_extensions:
                processed_documents.append({
                    "filename": uploaded_file.filename,
                    "status": "failed",
                    "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                })
                continue
            
            # Read file content into memory
            file_content = await uploaded_file.read()
            
            # Save file temporarily for processing
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            
            print(f"Processing file: {uploaded_file.filename}")
            
            # Process document: load and chunk
            chunks = document_service.process_document(file_path, file_extension)
            
            # Insert document metadata AND binary data into database
            document_id = db.insert_document(
                uploaded_file.filename, 
                file_extension,
                file_content  # Store actual file binary
            )
            
            # Generate embeddings and store chunks
            rag_service.embed_and_store_chunks(document_id, chunks)
            
            # Clean up temporary file
            os.remove(file_path)
            
            processed_documents.append({
                "filename": uploaded_file.filename,
                "document_id": document_id,
                "chunks_count": len(chunks),
                "file_size": len(file_content),
                "status": "success"
            })
            
            total_chunks += len(chunks)
            
        except ValueError as e:
            processed_documents.append({
                "filename": uploaded_file.filename,
                "status": "failed",
                "error": str(e)
            })
        except Exception as e:
            print(f"Error processing {uploaded_file.filename}: {e}")
            processed_documents.append({
                "filename": uploaded_file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    # Check if any documents were successfully processed
    successful_docs = [doc for doc in processed_documents if doc.get("status") == "success"]
    
    if not successful_docs:
        raise HTTPException(
            status_code=400, 
            detail="No documents were successfully processed",
            headers={"X-Details": str(processed_documents)}
        )
    
    return UploadResponse(
        message=f"Successfully processed {len(successful_docs)} out of {len(files)} documents",
        documents=processed_documents,
        total_documents=len(successful_docs),
        total_chunks=total_chunks
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about uploaded document(s)
    
    Args:
        request: Chat request with question and optional document_id
        
    Returns:
        Answer with source chunks
    """
    try:
        # Validate question
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # If no document_id provided, search across ALL documents
        document_id = request.document_id
        
        # Check if any documents exist
        if document_id is None:
            latest_id = db.get_latest_document_id()
            if latest_id is None:
                raise HTTPException(
                    status_code=404, 
                    detail="No documents found. Please upload a document first."
                )
        
        # Perform RAG query
        result = rag_service.query(
            question=request.question,
            document_id=document_id,  # None = search all documents
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
    """Download the original document file from database"""
    try:
        doc_data = db.get_document_file(document_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create file stream from binary data
        file_stream = BytesIO(bytes(doc_data['file_data']))
        
        # Determine media type
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