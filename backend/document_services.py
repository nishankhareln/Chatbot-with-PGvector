from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from typing import List


class DocumentService:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initialize document service with chunking parameters
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str, file_type: str) -> str:
        """
        Load document content based on file type
        
        Args:
            file_path: Path to the uploaded file
            file_type: Type of file (pdf, txt, md)
            
        Returns:
            Extracted text content
        """
        try:
            if file_type.lower() == 'pdf':
                return self._load_pdf(file_path)
            elif file_type.lower() in ['txt', 'md', 'markdown']:
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")
    
    def _load_pdf(self, file_path: str) -> str:
        """Load PDF file using PyPDFLoader"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # Combine all pages into single text
        text = "\n\n".join([doc.page_content for doc in documents])
        return text
    
    def _load_text(self, file_path: str) -> str:
        """Load text file (TXT or Markdown)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def process_document(self, file_path: str, file_type: str) -> List[str]:
        """
        Complete document processing pipeline: load and chunk
        
        Args:
            file_path: Path to the uploaded file
            file_type: Type of file (pdf, txt, md)
            
        Returns:
            List of text chunks
        """
        # Load document
        text = self.load_document(file_path, file_type)
        
        # Validate document is not empty
        if not text or len(text.strip()) == 0:
            raise ValueError("Document is empty or could not be read")
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        print(f"Document processed: {len(text)} characters -> {len(chunks)} chunks")
        return chunks
    
    def get_file_type(self, filename: str) -> str:
        """Extract file type from filename"""
        return filename.split('.')[-1].lower()


# Singleton instance
document_service = DocumentService()