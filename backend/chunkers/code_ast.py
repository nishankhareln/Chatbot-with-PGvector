"""
AST-based chunking for source code.

  - Python (.py): native `ast` module — splits by top-level functions / classes
    and walks into class bodies for methods.
  - Other languages (.js .ts .tsx .jsx .go .java .rs .cpp .c .rb .php):
    tree-sitter via tree_sitter_language_pack. Falls back to text chunking
    if the grammar isn't available at runtime.

Each chunk is annotated with the qualified symbol path (e.g. "ClassA.method_b")
so retrieval can show "where this code lives".
"""
import ast
import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import Chunk

LANG_BY_EXT = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "go": "go",
    "java": "java",
    "rs": "rust",
    "cpp": "cpp",
    "cc": "cpp",
    "c": "c",
    "h": "c",
    "rb": "ruby",
    "php": "php",
    "cs": "c_sharp",
    "kt": "kotlin",
    "swift": "swift",
}

CODE_NODE_TYPES = {
    "function_declaration",
    "function_definition",
    "method_definition",
    "method_declaration",
    "class_declaration",
    "class_definition",
    "interface_declaration",
    "type_alias_declaration",
    "arrow_function",
    "lexical_declaration",
}


class CodeASTChunker:
    def __init__(self, max_chunk_chars: int = 1800, fallback_chunk_size: int = 800):
        self.max_chunk_chars = max_chunk_chars
        self.fallback = RecursiveCharacterTextSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=120,
            separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""],
        )

    def chunk(self, text: str, file_ext: str, filename: str = "") -> List[Chunk]:
        ext = file_ext.lower().lstrip(".")
        if ext == "py":
            return self._chunk_python(text, filename)
        if ext in LANG_BY_EXT:
            return self._chunk_treesitter(text, ext, filename)
        return self._fallback(text)

    def _fallback(self, text: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        for i, piece in enumerate(self.splitter_split(text)):
            chunks.append(Chunk(text=piece, index=i, chunk_type="code"))
        return chunks

    def splitter_split(self, text: str) -> List[str]:
        return [c for c in self.fallback.split_text(text) if c.strip()]

    # ---------- Python via built-in ast ----------
    def _chunk_python(self, text: str, filename: str) -> List[Chunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._fallback(text)

        lines = text.splitlines()
        chunks: List[Chunk] = []
        idx = 0

        module_imports = self._collect_python_imports(tree, lines)
        if module_imports:
            chunks.append(
                Chunk(
                    text=module_imports,
                    index=idx,
                    parent_section="<module imports>",
                    chunk_type="code",
                    metadata={"filename": filename},
                )
            )
            idx += 1

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                src = self._slice_node(lines, node)
                for piece in self._split_if_too_big(src):
                    chunks.append(
                        Chunk(
                            text=piece,
                            index=idx,
                            parent_section=node.name,
                            chunk_type="code",
                            metadata={"symbol": node.name, "kind": "function"},
                        )
                    )
                    idx += 1
            elif isinstance(node, ast.ClassDef):
                class_header = self._slice_class_header(lines, node)
                if class_header:
                    chunks.append(
                        Chunk(
                            text=class_header,
                            index=idx,
                            parent_section=node.name,
                            chunk_type="code",
                            metadata={"symbol": node.name, "kind": "class"},
                        )
                    )
                    idx += 1
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        src = self._slice_node(lines, sub)
                        qual = f"{node.name}.{sub.name}"
                        for piece in self._split_if_too_big(src):
                            chunks.append(
                                Chunk(
                                    text=piece,
                                    index=idx,
                                    parent_section=qual,
                                    chunk_type="code",
                                    metadata={"symbol": qual, "kind": "method"},
                                )
                            )
                            idx += 1

        if not chunks:
            return self._fallback(text)
        return chunks

    def _collect_python_imports(self, tree: ast.Module, lines: List[str]) -> str:
        out: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out.append(self._slice_node(lines, node))
        return "\n".join(out).strip()

    def _slice_node(self, lines: List[str], node: ast.AST) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(lines[start:end])

    def _slice_class_header(self, lines: List[str], node: ast.ClassDef) -> Optional[str]:
        first_method_line = None
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_method_line = sub.lineno - 1
                break
        start = node.lineno - 1
        end = first_method_line if first_method_line is not None else node.end_lineno
        header = "\n".join(lines[start:end]).rstrip()
        return header or None

    def _split_if_too_big(self, src: str) -> List[str]:
        if len(src) <= self.max_chunk_chars:
            return [src]
        return self.splitter_split(src)

    # ---------- Other languages via tree-sitter ----------
    def _chunk_treesitter(self, text: str, ext: str, filename: str) -> List[Chunk]:
        try:
            from tree_sitter_language_pack import get_parser
        except Exception:
            return self._fallback(text)

        lang_name = LANG_BY_EXT.get(ext)
        if not lang_name:
            return self._fallback(text)

        try:
            parser = get_parser(lang_name)
        except Exception:
            return self._fallback(text)

        tree = parser.parse(text.encode("utf-8"))
        root = tree.root_node
        chunks: List[Chunk] = []
        idx = 0
        source_bytes = text.encode("utf-8")

        def walk(node, parent_path: str = ""):
            nonlocal idx
            for child in node.children:
                if child.type in CODE_NODE_TYPES:
                    snippet = source_bytes[child.start_byte:child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    name = self._extract_name(child, source_bytes) or child.type
                    qual = f"{parent_path}.{name}" if parent_path else name
                    for piece in self._split_if_too_big(snippet):
                        chunks.append(
                            Chunk(
                                text=piece,
                                index=idx,
                                parent_section=qual,
                                chunk_type="code",
                                metadata={"symbol": qual, "kind": child.type},
                            )
                        )
                        idx += 1
                    if child.type in {
                        "class_declaration",
                        "class_definition",
                        "interface_declaration",
                    }:
                        walk(child, qual)
                else:
                    walk(child, parent_path)

        walk(root)
        if not chunks:
            return self._fallback(text)
        return chunks

    def _extract_name(self, node, source_bytes: bytes) -> Optional[str]:
        for child in node.children:
            if child.type in {"identifier", "type_identifier", "property_identifier"}:
                return source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )
        return None
