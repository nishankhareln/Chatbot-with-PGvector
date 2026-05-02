"""
Hierarchical / structure-aware chunking for documents.

Strategy:
  1. Detect headings (markdown # / ##, PDF "Page N" markers, ALL-CAPS lines).
  2. Build a tree of sections (path = "H1 > H2 > H3").
  3. Within each section, split by RecursiveCharacterTextSplitter respecting
     the configured chunk_size / overlap.
  4. Each chunk carries its full section path as parent_section metadata,
     which is later used for context injection and BM25 weighting.
"""
import re
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import Chunk

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
PAGE_MARKER = re.compile(r"^---\s*Page\s+(\d+).*?---\s*$", re.MULTILINE | re.IGNORECASE)
ALL_CAPS_HEADING = re.compile(r"^([A-Z][A-Z0-9 \-:_,]{4,80})\s*$", re.MULTILINE)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-Z][^\n]{2,100})\s*$", re.MULTILINE)


class HierarchicalChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _detect_sections(self, text: str) -> List[Tuple[int, int, str]]:
        """Return list of (start_offset, level, title) sorted by offset."""
        sections: List[Tuple[int, int, str]] = []

        for m in MD_HEADING.finditer(text):
            level = len(m.group(1))
            sections.append((m.start(), level, m.group(2).strip()))

        for m in NUMBERED_HEADING.finditer(text):
            level = m.group(1).count(".") + 1
            title = f"{m.group(1)} {m.group(2).strip()}"
            sections.append((m.start(), level, title))

        for m in PAGE_MARKER.finditer(text):
            sections.append((m.start(), 9, f"Page {m.group(1)}"))

        if len(sections) < 3:
            for m in ALL_CAPS_HEADING.finditer(text):
                line = m.group(1).strip()
                if 1 <= len(line.split()) <= 12:
                    sections.append((m.start(), 2, line))

        sections.sort(key=lambda s: s[0])
        deduped: List[Tuple[int, int, str]] = []
        for s in sections:
            if deduped and s[0] == deduped[-1][0]:
                continue
            deduped.append(s)
        return deduped

    def _build_section_path(
        self, sections: List[Tuple[int, int, str]], offset: int
    ) -> str:
        path: List[Tuple[int, str]] = []
        for start, level, title in sections:
            if start > offset:
                break
            path = [(lvl, t) for lvl, t in path if lvl < level]
            path.append((level, title))
        return " > ".join(t for _, t in path) if path else ""

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sections = self._detect_sections(text)
        chunks: List[Chunk] = []
        idx = 0

        if not sections:
            for piece in self.splitter.split_text(text):
                chunks.append(Chunk(text=piece, index=idx, parent_section=""))
                idx += 1
            return chunks

        boundaries = [s[0] for s in sections] + [len(text)]
        for i in range(len(sections)):
            start = boundaries[i]
            end = boundaries[i + 1]
            block = text[start:end].strip()
            if not block:
                continue
            section_path = self._build_section_path(sections, start)

            for piece in self.splitter.split_text(block):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        text=piece,
                        index=idx,
                        parent_section=section_path,
                        chunk_type="text",
                    )
                )
                idx += 1

        return chunks
