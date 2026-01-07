from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from typing import List, Dict
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\LOQ\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111 (1).exe"


class AdvancedDocumentService:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        """
        Advanced document processing with multiple extraction methods
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
        """Load document with automatic method selection"""
        try:
            if file_type.lower() == 'pdf':
                return self._load_pdf_multimethod(file_path)
            elif file_type.lower() in ['txt', 'md', 'markdown']:
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")
    
    def _load_pdf_multimethod(self, file_path: str) -> str:
        """
        Try multiple PDF extraction methods in order of preference
        """
        print(f"\n{'='*60}")
        print(f"📄 ADVANCED PDF PROCESSING: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        extracted_text = ""
        methods_tried = []
        
        # METHOD 1: PyPDF2 (Fast, basic extraction)
        print("\nMethod 1: PyPDF2 (Basic Text Extraction)")
        try:
            text = self._extract_with_pypdf2(file_path)
            methods_tried.append(("PyPDF2", len(text) if text else 0))
            
            if text and len(text.strip()) > 100:  # At least 100 chars
                print(f"  SUCCESS: Extracted {len(text)} characters")
                extracted_text = text
            else:
                print(f" Insufficient text: {len(text) if text else 0} chars")
        except Exception as e:
            print(f"  FAILED: {str(e)[:100]}")
            methods_tried.append(("PyPDF2", 0))
        
        # If PyPDF2 worked well, return
        if len(extracted_text) > 100:
            return self._clean_text(extracted_text)
        
        # METHOD 2: pdfplumber (Better for tables and complex layouts)
        print("\n🔍 Method 2: pdfplumber (Advanced Extraction)")
        try:
            text = self._extract_with_pdfplumber(file_path)
            methods_tried.append(("pdfplumber", len(text) if text else 0))
            
            if text and len(text.strip()) > 100:
                print(f"  SUCCESS: Extracted {len(text)} characters")
                extracted_text = text
            else:
                print(f"    Insufficient text: {len(text) if text else 0} chars")
        except Exception as e:
            print(f"    FAILED: {str(e)[:100]}")
            methods_tried.append(("pdfplumber", 0))
        
        # If pdfplumber worked, return
        if len(extracted_text) > 100:
            return self._clean_text(extracted_text)
        
        # METHOD 3: OCR (For scanned/image PDFs)
        print("\n🔍 Method 3: OCR (Optical Character Recognition)")
        print("   ⚠️  This may take 30-60 seconds for scanned PDFs...")
        try:
            text = self._extract_with_ocr(file_path)
            methods_tried.append(("OCR", len(text) if text else 0))
            
            if text and len(text.strip()) > 100:
                print(f"   ✅ SUCCESS: Extracted {len(text)} characters via OCR")
                extracted_text = text
            else:
                print(f"   ⚠️  Insufficient text: {len(text) if text else 0} chars")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)[:100]}")
            methods_tried.append(("OCR", 0))
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 EXTRACTION SUMMARY:")
        for method, chars in methods_tried:
            status = "✅" if chars > 100 else "❌"
            print(f"   {status} {method}: {chars} characters")
        print(f"{'='*60}\n")
        
        # Final check
        if not extracted_text or len(extracted_text.strip()) < 100:
            raise Exception(
                f"Could not extract sufficient text from PDF. "
                f"Tried {len(methods_tried)} methods. "
                f"This PDF might be: (1) Empty, (2) Heavily encrypted, "
                f"(3) Pure images without text, or (4) Corrupted."
            )
        
        return self._clean_text(extracted_text)
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n\n"
                    text += page_text
            
            return text
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber (better for tables)"""
        text = ""
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n\n"
                    text += page_text
                
                # Extract tables separately
                tables = page.extract_tables()
                if tables:
                    text += "\n\n[TABLES ON THIS PAGE]:\n"
                    for table_num, table in enumerate(tables):
                        text += f"\nTable {table_num + 1}:\n"
                        for row in table:
                            text += " | ".join([str(cell) if cell else "" for cell in row])
                            text += "\n"
        
        return text
    
    def _extract_with_ocr(self, file_path: str, max_pages: int = 10) -> str:
        """
        Extract text using OCR (for scanned PDFs)
        
        Args:
            file_path: Path to PDF
            max_pages: Maximum pages to OCR (to avoid timeout)
        """
        text = ""
        
        try:
            # Convert PDF pages to images
            print(f"   📸 Converting PDF to images...")
            images = convert_from_path(
                file_path, 
                dpi=300,  # Higher DPI = better OCR quality
                first_page=1,
                last_page=max_pages  # Limit pages to avoid timeout
            )
            
            print(f"   🔤 Running OCR on {len(images)} pages...")
            
            # OCR each page
            for page_num, image in enumerate(images):
                print(f"      Processing page {page_num + 1}/{len(images)}...")
                
                # Perform OCR
                page_text = pytesseract.image_to_string(
                    image,
                    lang='eng',  # Add more languages if needed: 'eng+fra+deu'
                    config='--psm 1'  # Page segmentation mode: automatic
                )
                
                if page_text.strip():
                    text += f"\n\n--- Page {page_num + 1} (OCR) ---\n\n"
                    text += page_text
            
            return text
            
        except Exception as e:
            raise Exception(f"OCR failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = '\n'.join([line.strip() for line in text.split('\n')])
        
        # Remove multiple blank lines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        # Remove special characters that cause issues
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')
        
        return text.strip()
    
    def _load_text(self, file_path: str) -> str:
        """Load text file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                return text
            except UnicodeDecodeError:
                continue
        
        raise Exception(f"Could not decode text file with any encoding: {encodings}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def process_document(self, file_path: str, file_type: str) -> List[str]:
        """
        Complete document processing pipeline
        """
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("File is empty (0 bytes)")
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB. Maximum 50MB.")
        
        # Load document
        text = self.load_document(file_path, file_type)
        
        # Validate
        if not text or len(text.strip()) < 50:
            raise ValueError(
                f"Could not extract meaningful text. "
                f"Extracted only {len(text.strip())} characters. "
                f"Minimum required: 50 characters."
            )
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        if not chunks or len(chunks) == 0:
            raise ValueError("Failed to create chunks from extracted text")
        
        print(f"\n✅ Document processed successfully:")
        print(f"   📝 Extracted: {len(text)} characters")
        print(f"   📦 Created: {len(chunks)} chunks")
        print(f"   📏 Chunk size: {self.chunk_size} chars")
        print(f"   🔗 Overlap: {self.chunk_overlap} chars\n")
        
        return chunks
    
    def get_file_type(self, filename: str) -> str:
        """Extract file type from filename"""
        return filename.split('.')[-1].lower()
    
    def detect_pdf_type(self, file_path: str) -> Dict:
        """
        Detect what type of PDF this is
        
        Returns dict with:
        - is_text_based: bool
        - is_scanned: bool
        - has_images: bool
        - page_count: int
        - estimated_method: str
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                
                # Try to extract text from first page
                first_page_text = reader.pages[0].extract_text()
                
                # Detect type
                is_text_based = len(first_page_text.strip()) > 100
                is_scanned = not is_text_based
                
                return {
                    "is_text_based": is_text_based,
                    "is_scanned": is_scanned,
                    "page_count": page_count,
                    "estimated_method": "PyPDF2" if is_text_based else "OCR",
                    "sample_text": first_page_text[:200] if is_text_based else None
                }
        except:
            return {
                "is_text_based": False,
                "is_scanned": True,
                "page_count": 0,
                "estimated_method": "Unknown"
            }


# Singleton instance
document_service = AdvancedDocumentService()