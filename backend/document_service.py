"""
Document service: extract text from a file, then dispatch to the correct
structure-aware chunker.

  - .pdf : multi-method extraction (PyPDF2 -> pdfplumber -> OCR) then
           HierarchicalChunker (sections / pages).
  - .md / .markdown / .txt : HierarchicalChunker (markdown headings /
                              ALL-CAPS / numbered).
  - .py / .js / .ts / .go / .java / .rs / ... : CodeASTChunker
                              (Python ast, others tree-sitter).
"""
from __future__ import annotations

import os
from typing import List

import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from chunkers import HierarchicalChunker, CodeASTChunker, Chunk

# Tesseract path: env-driven, with sensible fallbacks per platform.
_tess_env = os.getenv("TESSERACT_CMD")
if _tess_env and os.path.exists(_tess_env):
    pytesseract.pytesseract.tesseract_cmd = _tess_env
elif os.name == "nt":
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

CODE_EXTENSIONS = {
    "py", "js", "jsx", "ts", "tsx", "go", "java",
    "rs", "cpp", "cc", "c", "h", "rb", "php", "cs", "kt", "swift",
}
TEXT_EXTENSIONS = {"txt", "md", "markdown", "rst"}


class DocumentService:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hierarchical = HierarchicalChunker(chunk_size, chunk_overlap)
        self.code = CodeASTChunker(max_chunk_chars=chunk_size * 2)

    # -------------------------- public api --------------------------
    def process_document(self, file_path: str, file_type: str) -> List[Chunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        size = os.path.getsize(file_path)
        if size == 0:
            raise ValueError("File is empty (0 bytes)")
        if size > 50 * 1024 * 1024:
            raise ValueError(f"File too large: {size / 1_048_576:.1f}MB (limit 50MB)")

        ext = file_type.lower().lstrip(".")
        text = self._extract_text(file_path, ext)

        if not text or len(text.strip()) < 30:
            raise ValueError(
                f"Extracted only {len(text.strip())} characters — file may be empty, "
                "scanned without OCR support, or corrupted."
            )

        if ext in CODE_EXTENSIONS:
            chunks = self.code.chunk(text, ext, filename=os.path.basename(file_path))
        else:
            chunks = self.hierarchical.chunk(text)

        if not chunks:
            raise ValueError("Chunker produced no chunks")

        print(
            f"[document_service] {os.path.basename(file_path)} "
            f"-> {len(text)} chars -> {len(chunks)} chunks "
            f"({'code' if ext in CODE_EXTENSIONS else 'doc'})"
        )
        return chunks

    def get_file_type(self, filename: str) -> str:
        return filename.rsplit(".", 1)[-1].lower()

    # -------------------------- extraction --------------------------
    def _extract_text(self, file_path: str, ext: str) -> str:
        if ext == "pdf":
            return self._extract_pdf(file_path)
        if ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS:
            return self._read_text_file(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    def _read_text_file(self, file_path: str) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {file_path}")

    def _extract_pdf(self, file_path: str) -> str:
        text = self._pypdf2(file_path)
        if self._is_good(text):
            return self._clean(text)

        text = self._pdfplumber(file_path)
        if self._is_good(text):
            return self._clean(text)

        text = self._ocr(file_path)
        if self._is_good(text):
            return self._clean(text)

        raise ValueError(
            "Could not extract sufficient text from PDF (tried PyPDF2, pdfplumber, OCR)"
        )

    @staticmethod
    def _is_good(text: str) -> bool:
        return bool(text) and len(text.strip()) > 100

    def _pypdf2(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                parts = []
                for i, page in enumerate(reader.pages):
                    t = page.extract_text() or ""
                    if t.strip():
                        parts.append(f"\n\n--- Page {i + 1} ---\n\n{t}")
                return "".join(parts)
        except Exception as e:
            print(f"[pypdf2] failed: {e}")
            return ""

    def _pdfplumber(self, file_path: str) -> str:
        try:
            parts = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text() or ""
                    if t.strip():
                        parts.append(f"\n\n--- Page {i + 1} ---\n\n{t}")
                    tables = page.extract_tables() or []
                    for j, table in enumerate(tables):
                        rows = "\n".join(
                            " | ".join((c or "").strip() for c in row) for row in table
                        )
                        parts.append(f"\n[Table {j + 1} — Page {i + 1}]\n{rows}\n")
            return "".join(parts)
        except Exception as e:
            print(f"[pdfplumber] failed: {e}")
            return ""

    def _ocr(self, file_path: str, max_pages: int = 10) -> str:
        try:
            images = convert_from_path(
                file_path, dpi=300, first_page=1, last_page=max_pages
            )
            parts = []
            for i, img in enumerate(images):
                t = pytesseract.image_to_string(img, lang="eng", config="--psm 1")
                if t.strip():
                    parts.append(f"\n\n--- Page {i + 1} (OCR) ---\n\n{t}")
            return "".join(parts)
        except Exception as e:
            print(f"[ocr] failed: {e}")
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        if not text:
            return ""
        text = "\n".join(line.rstrip() for line in text.splitlines())
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.replace("\x00", "").replace("�", "").strip()


document_service = DocumentService()
