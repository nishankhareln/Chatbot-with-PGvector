from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from typing import Optional, List
from io import BytesIO
import traceback

from database import db
from document_services import document_service
from rag_service import rag_service

# Create FastAPI app
app = FastAPI(
    title="RAG Assistant API",
    description="Document Q&A System with RAG Architecture",
    version="1.0.0"
)

# CORS middleware - allow all origins for development
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


# ========================================
# PYDANTIC MODELS
# ========================================

class ChatRequest(BaseModel):
    question: str
    document_id: Optional[int] = None
    top_k: int = 3
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of the document?",
                "document_id": None,
                "top_k": 3
            }
        }


class SourceInfo(BaseModel):
    text: str
    filename: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The main topic is...",
                "sources": [
                    {
                        "text": "Relevant text chunk...",
                        "filename": "document.pdf",
                        "similarity": 0.85
                    }
                ]
            }
        }


class DocumentInfo(BaseModel):
    filename: str
    document_id: int
    chunks_count: int
    file_size: int
    status: str


class UploadResponse(BaseModel):
    message: str
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentListItem(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    uploaded_at: str
    chunk_count: Optional[int] = 0


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None


# ========================================
# ROOT & HEALTH ENDPOINTS
# ========================================

@app.get("/")
def root():
    """
    Root endpoint - API documentation
    """
    return {
        "name": "RAG Assistant API",
        "version": "1.0.0",
        "description": "Document Q&A System with Retrieval Augmented Generation",
        "endpoints": {
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Check API and database health"
            },
            "upload": {
                "path": "/upload",
                "method": "POST",
                "description": "Upload multiple documents (PDF, TXT, MD)",
                "accepts": "multipart/form-data"
            },
            "chat": {
                "path": "/chat",
                "method": "POST",
                "description": "Ask questions about documents",
                "accepts": "application/json"
            },
            "documents": {
                "path": "/documents",
                "method": "GET",
                "description": "List all uploaded documents"
            },
            "document_info": {
                "path": "/document/{id}",
                "method": "GET",
                "description": "Get document details"
            },
            "download": {
                "path": "/document/{id}/download",
                "method": "GET",
                "description": "Download original document"
            },
            "delete": {
                "path": "/document/{id}",
                "method": "DELETE",
                "description": "Delete document and chunks"
            }
        },
        "status": "operational"
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    Verifies database connection and system status
    """
    try:
        db_healthy = db.health_check()
        
        # Get document count as additional health metric
        doc_count = 0
        try:
            documents = db.get_all_documents()
            doc_count = len(documents)
        except:
            pass
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "documents_count": doc_count,
            "api_version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e)
        }


# ========================================
# DOCUMENT UPLOAD ENDPOINT
# ========================================

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple documents (PDF, TXT, or Markdown)
    
    **Process:**
    1. Validates file types
    2. Extracts text content
    3. Splits into chunks
    4. Generates embeddings
    5. Stores in vector database
    
    **Args:**
    - files: List of uploaded files (PDF, TXT, MD)
    
    **Returns:**
    - Upload status with document IDs and chunk counts
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_documents = []
    total_chunks = 0
    allowed_extensions = ['pdf', 'txt', 'md', 'markdown']
    
    for uploaded_file in files:
        try:
            # Validate file type
            file_extension = uploaded_file.filename.split('.')[-1].lower()
            
            if file_extension not in allowed_extensions:
                processed_documents.append(DocumentInfo(
                    filename=uploaded_file.filename,
                    document_id=-1,
                    chunks_count=0,
                    file_size=0,
                    status=f"failed: Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                ))
                continue
            
            # Validate file size (max 50MB)
            file_content = await uploaded_file.read()
            file_size = len(file_content)
            
            if file_size == 0:
                processed_documents.append(DocumentInfo(
                    filename=uploaded_file.filename,
                    document_id=-1,
                    chunks_count=0,
                    file_size=0,
                    status="failed: Empty file"
                ))
                continue
            
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                processed_documents.append(DocumentInfo(
                    filename=uploaded_file.filename,
                    document_id=-1,
                    chunks_count=0,
                    file_size=file_size,
                    status="failed: File too large (max 50MB)"
                ))
                continue
            
            # Save file temporarily for processing
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            
            print(f"📄 Processing file: {uploaded_file.filename} ({file_size} bytes)")
            
            # Process document: extract text and chunk
            chunks = document_service.process_document(file_path, file_extension)
            
            if not chunks or len(chunks) == 0:
                os.remove(file_path)
                processed_documents.append(DocumentInfo(
                    filename=uploaded_file.filename,
                    document_id=-1,
                    chunks_count=0,
                    file_size=file_size,
                    status="failed: No text content extracted"
                ))
                continue
            
            # Insert document metadata AND binary data into database
            document_id = db.insert_document(
                uploaded_file.filename, 
                file_extension,
                file_content
            )
            
            print(f"📝 Stored document with ID: {document_id}")
            
            # Generate embeddings and store chunks
            print(f"🔍 Generating embeddings for {len(chunks)} chunks...")
            rag_service.embed_and_store_chunks(document_id, chunks)
            
            # Clean up temporary file
            os.remove(file_path)
            
            processed_documents.append(DocumentInfo(
                filename=uploaded_file.filename,
                document_id=document_id,
                chunks_count=len(chunks),
                file_size=file_size,
                status="success"
            ))
            
            total_chunks += len(chunks)
            print(f"✅ Successfully processed: {uploaded_file.filename}")
            
        except ValueError as e:
            print(f"❌ ValueError processing {uploaded_file.filename}: {e}")
            processed_documents.append(DocumentInfo(
                filename=uploaded_file.filename,
                document_id=-1,
                chunks_count=0,
                file_size=0,
                status=f"failed: {str(e)}"
            ))
        except Exception as e:
            print(f"❌ Error processing {uploaded_file.filename}: {e}")
            print(traceback.format_exc())
            processed_documents.append(DocumentInfo(
                filename=uploaded_file.filename,
                document_id=-1,
                chunks_count=0,
                file_size=0,
                status=f"failed: {str(e)}"
            ))
            # Clean up temp file if it exists
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # Check if any documents were successfully processed
    successful_docs = [doc for doc in processed_documents if doc.status == "success"]
    
    if not successful_docs:
        raise HTTPException(
            status_code=400, 
            detail={
                "message": "No documents were successfully processed",
                "details": [{"filename": doc.filename, "status": doc.status} for doc in processed_documents]
            }
        )
    
    return UploadResponse(
        message=f"Successfully processed {len(successful_docs)} out of {len(files)} documents",
        documents=processed_documents,
        total_documents=len(successful_docs),
        total_chunks=total_chunks
    )


# ========================================
# CHAT ENDPOINT
# ========================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about uploaded document(s)
    
    **Process:**
    1. Validates question
    2. Generates query embedding
    3. Retrieves top-k similar chunks
    4. Constructs context from chunks
    5. Generates answer using LLM
    
    **Args:**
    - question: User's question
    - document_id: Optional - specific document to search (None = all documents)
    - top_k: Number of relevant chunks to retrieve (default: 3)
    
    **Returns:**
    - AI-generated answer with source chunks
    """
    try:
        # Validate question
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(request.question) > 2000:
            raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters)")
        
        # Validate top_k
        if request.top_k < 1 or request.top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
        
        document_id = request.document_id
        
        # Check if any documents exist
        if document_id is None:
            latest_id = db.get_latest_document_id()
            if latest_id is None:
                raise HTTPException(
                    status_code=404, 
                    detail="No documents found. Please upload a document first."
                )
        else:
            # Verify document exists
            doc_info = db.get_document_info(document_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document with ID {document_id} not found"
                )
        
        print(f"💬 Query: {request.question}")
        print(f"📚 Document ID: {document_id if document_id else 'All documents'}")
        print(f"🔢 Top K: {request.top_k}")
        
        # Perform RAG query
        result = rag_service.query(
            question=request.question,
            document_id=document_id,
            top_k=request.top_k
        )
        
        print(f"✅ Generated answer with {len(result['sources'])} sources")
        
        return ChatResponse(
            answer=result["answer"],
            sources=[
                SourceInfo(
                    text=source["text"],
                    filename=source["filename"],
                    similarity=source["similarity"]
                )
                for source in result["sources"]
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating answer: {str(e)}"
        )


# ========================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ========================================

@app.get("/documents")
def list_documents():
    """
    Get list of all uploaded documents with metadata
    
    **Returns:**
    - List of documents with ID, filename, type, size, and upload date
    """
    try:
        documents = db.get_all_documents()
        
        # Format response
        formatted_docs = []
        for doc in documents:
            formatted_docs.append(DocumentListItem(
                id=doc["id"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                file_size=doc["file_size"],
                uploaded_at=doc["uploaded_at"],
                chunk_count=doc.get("chunk_count", 0)
            ))
        
        return {
            "documents": formatted_docs,
            "count": len(formatted_docs)
        }
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{document_id}")
def get_document_info(document_id: int):
    """
    Get detailed information about a specific document
    
    **Args:**
    - document_id: ID of the document
    
    **Returns:**
    - Document metadata including chunk count and statistics
    """
    try:
        doc_info = db.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Document with ID {document_id} not found"
            )
        return doc_info
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{document_id}/download")
def download_document(document_id: int):
    """
    Download the original document file
    
    **Args:**
    - document_id: ID of the document
    
    **Returns:**
    - File stream with appropriate content type
    """
    try:
        doc_data = db.get_document_file(document_id)
        if not doc_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Document with ID {document_id} not found"
            )
        
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
        
        print(f"📥 Downloading: {doc_data['filename']}")
        
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
        print(f"❌ Error downloading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{document_id}")
def delete_document(document_id: int):
    """
    Delete a document and all its associated chunks
    
    **Args:**
    - document_id: ID of the document to delete
    
    **Returns:**
    - Success message
    """
    try:
        # Verify document exists
        doc_info = db.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {document_id} not found"
            )
        
        filename = doc_info.get("filename", "unknown")
        
        # Delete document and chunks
        db.delete_document(document_id)
        
        print(f"🗑️ Deleted document: {filename} (ID: {document_id})")
        
        return {
            "message": f"Document '{filename}' deleted successfully",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# ERROR HANDLERS
# ========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    print(f"❌ Unhandled exception: {exc}")
    print(traceback.format_exc())
    return {
        "detail": "Internal server error",
        "error": str(exc)
    }


# ========================================
# STARTUP & SHUTDOWN EVENTS
# ========================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("🚀 RAG Assistant API Starting...")
    print("=" * 60)
    
    # Check database connection
    if db.health_check():
        print("✅ Database connected")
    else:
        print("⚠️ Database connection failed")
    
    # Check upload directory
    if os.path.exists(UPLOAD_DIR):
        print(f"✅ Upload directory ready: {UPLOAD_DIR}")
    else:
        print(f"⚠️ Creating upload directory: {UPLOAD_DIR}")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    print("=" * 60)
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("=" * 60)
    print("👋 RAG Assistant API Shutting Down...")
    print("=" * 60)


# ========================================
# MAIN ENTRY POINT
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting RAG Assistant API...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )