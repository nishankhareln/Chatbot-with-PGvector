"""
Advanced RAG pipeline.

Ingest:
  embed(chunk + parent_section + context_summary) using BAAI/bge-small-en-v1.5
  with Anthropic-style **contextual retrieval**: a Gemini-Flash pass adds a
  one-sentence "where this fits in the doc" line to each chunk before embedding.

Query:
  1. Optional query rewrite (LLM expands acronyms / fragments).
  2. Optional HyDE (LLM writes a hypothetical answer, embed *that*) for short
     queries — drastically improves recall on under-specified questions.
  3. Hybrid retrieval (dense pgvector HNSW + BM25 tsvector) fused with RRF.
  4. Cross-encoder reranker (BAAI/bge-reranker-base) on the fused candidates.
  5. Sentence-window expansion: include neighbour chunks for richer context.
  6. Chat-history-aware prompt with citation enforcement, streamed back token
     by token.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from database import db
from chunkers import Chunk

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
USE_CONTEXTUAL = os.getenv("USE_CONTEXTUAL_RETRIEVAL", "1") == "1"


class RAGService:
    def __init__(self) -> None:
        print(f"[rag] loading embedder: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.reranker = None
        if USE_RERANKER:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[rag] loading reranker: {RERANKER_MODEL}")
                self.reranker = CrossEncoder(RERANKER_MODEL)
            except Exception as e:
                print(f"[rag] reranker disabled ({e})")

        if not GEMINI_API_KEY:
            print("[rag] WARNING: GEMINI_API_KEY not set — generation will fail")
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        print("[rag] ready")

    # ============================ embeddings ============================
    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self.embedder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )

    @lru_cache(maxsize=512)
    def _cached_query_embed(self, text: str) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.embed(text))

    def query_embed(self, text: str) -> np.ndarray:
        return np.array(self._cached_query_embed(text))

    # ============================ ingest ============================
    def embed_and_store_chunks(self, document_id: int, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        full_doc_preview = self._doc_preview(chunks)
        contexts: List[Optional[str]] = [None] * len(chunks)
        if USE_CONTEXTUAL and self.client:
            print(f"[rag] generating contextual summaries for {len(chunks)} chunks")
            for i, ch in enumerate(chunks):
                try:
                    contexts[i] = self._chunk_context(full_doc_preview, ch.text)
                except Exception as e:
                    print(f"[rag] context gen failed on chunk {i}: {e}")

        embed_inputs = [
            self._compose_for_embedding(ch, contexts[i]) for i, ch in enumerate(chunks)
        ]
        print(f"[rag] embedding {len(embed_inputs)} chunks")
        embeddings = self.embed_batch(embed_inputs)

        rows = [
            (
                ch.text,
                ch.index,
                emb.tolist(),
                ch.parent_section or None,
                contexts[i],
                ch.chunk_type or "text",
            )
            for i, (ch, emb) in enumerate(zip(chunks, embeddings))
        ]
        db.insert_chunks_batch(document_id, rows)
        print(f"[rag] stored {len(rows)} chunks for document {document_id}")

    def _compose_for_embedding(self, ch: Chunk, context: Optional[str]) -> str:
        parts: List[str] = []
        if ch.parent_section:
            parts.append(f"Section: {ch.parent_section}")
        if context:
            parts.append(f"Context: {context}")
        parts.append(ch.text)
        return "\n".join(parts)

    def _doc_preview(self, chunks: List[Chunk], max_chars: int = 3000) -> str:
        joined = "\n".join(c.text for c in chunks)
        return joined[:max_chars]

    def _chunk_context(self, doc_preview: str, chunk_text: str) -> str:
        prompt = (
            "<document>\n"
            f"{doc_preview}\n"
            "</document>\n"
            "Here is a chunk from this document:\n"
            "<chunk>\n"
            f"{chunk_text[:1500]}\n"
            "</chunk>\n"
            "Give a single short sentence (max 25 words) situating this chunk inside "
            "the overall document so it can be retrieved on its own. "
            "Reply with the sentence only — no preamble."
        )
        resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip().split("\n")[0][:300]

    # ============================ query rewriting + HyDE ============================
    def _is_short(self, q: str) -> bool:
        return len(q.split()) <= 4

    def _rewrite_query(self, query: str, history: List[Dict]) -> str:
        if not self.client:
            return query
        history_str = "\n".join(
            f"{m['role']}: {m['content']}" for m in history[-4:]
        ) or "(no prior turns)"
        prompt = (
            "Rewrite the user's latest question into a self-contained search query. "
            "Expand acronyms, resolve pronouns using the conversation, and add 1-2 "
            "relevant synonyms. Output ONLY the rewritten query.\n\n"
            f"Conversation so far:\n{history_str}\n\n"
            f"Latest question: {query}\n\nRewritten query:"
        )
        try:
            resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            out = (resp.text or "").strip().splitlines()[0]
            return out or query
        except Exception as e:
            print(f"[rag] rewrite failed: {e}")
            return query

    def _hyde(self, query: str) -> Optional[str]:
        if not self.client:
            return None
        prompt = (
            "Write a short, plausible 2-3 sentence answer to the question below as if "
            "it appeared in a reference document. Be specific and use domain vocabulary. "
            "Do not say 'I don't know'. Output the passage only.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        try:
            resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return (resp.text or "").strip()
        except Exception as e:
            print(f"[rag] hyde failed: {e}")
            return None

    # ============================ retrieval ============================
    def retrieve(
        self,
        query: str,
        document_id: Optional[int] = None,
        top_k: int = 5,
        history: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Dict]:
        history = history or []
        diagnostics: Dict = {"original_query": query}

        rewritten = self._rewrite_query(query, history) if history or self._is_short(query) else query
        diagnostics["rewritten_query"] = rewritten

        # Choose what to embed: HyDE for short queries, otherwise the rewritten query.
        embed_text = rewritten
        if self._is_short(query):
            hyde = self._hyde(rewritten)
            if hyde:
                embed_text = hyde
                diagnostics["hyde"] = hyde

        q_emb = self.query_embed(embed_text)

        candidates = db.hybrid_search(
            query_embedding=q_emb,
            query_text=rewritten,
            top_k=max(top_k * 4, 20),
            document_id=document_id,
        )
        diagnostics["candidates"] = len(candidates)

        # Rerank
        if self.reranker and candidates:
            pairs = [(rewritten, c["chunk_text"]) for c in candidates]
            try:
                scores = self.reranker.predict(pairs)
                for c, s in zip(candidates, scores):
                    c["rerank_score"] = float(s)
                candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
                diagnostics["reranked"] = True
            except Exception as e:
                print(f"[rag] rerank failed: {e}")

        top = candidates[:top_k]

        # Sentence-window: enrich with neighbours
        enriched: List[Dict] = []
        seen_ids = set()
        for c in top:
            neighbours = db.get_neighbors(c["document_id"], c["chunk_index"], window=1)
            window_text = "\n".join(n["chunk_text"] for n in neighbours)
            c["window_text"] = window_text or c["chunk_text"]
            if c["id"] not in seen_ids:
                enriched.append(c)
                seen_ids.add(c["id"])

        return enriched, diagnostics

    # ============================ generation ============================
    def _build_prompt(
        self, question: str, chunks: List[Dict], history: List[Dict]
    ) -> str:
        if not chunks:
            return (
                "You have no document context. Tell the user no relevant content was "
                f"found and ask them to upload or rephrase.\n\nQuestion: {question}"
            )

        ctx_blocks = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] (source: {c['filename']}"
            if c.get("parent_section"):
                header += f" — {c['parent_section']}"
            header += ")"
            body = c.get("window_text") or c["chunk_text"]
            ctx_blocks.append(f"{header}\n{body}")
        context = "\n\n".join(ctx_blocks)

        history_str = ""
        if history:
            history_str = "Conversation so far:\n" + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
            ) + "\n\n"

        return (
            "You are a careful research assistant answering ONLY from the provided "
            "context. If the context does not contain the answer, say so explicitly.\n\n"
            "Rules:\n"
            "- Cite sources inline using [1], [2], ... matching the context block numbers.\n"
            "- Be concise but specific. Prefer bullet points for multi-part answers.\n"
            "- If the user is asking a follow-up, use the conversation history for "
            "pronoun/antecedent resolution.\n"
            "- Never invent facts not present in the context.\n\n"
            f"{history_str}"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    def generate(self, question: str, chunks: List[Dict], history: List[Dict]) -> str:
        if not self.client:
            return "Generation unavailable: GEMINI_API_KEY not configured."
        prompt = self._build_prompt(question, chunks, history)
        try:
            resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return resp.text or ""
        except Exception as e:
            return f"Error generating answer: {e}"

    def generate_stream(
        self, question: str, chunks: List[Dict], history: List[Dict]
    ) -> Iterable[str]:
        if not self.client:
            yield "Generation unavailable: GEMINI_API_KEY not configured."
            return
        prompt = self._build_prompt(question, chunks, history)
        try:
            stream = self.client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt
            )
            for ev in stream:
                txt = getattr(ev, "text", None)
                if txt:
                    yield txt
        except Exception as e:
            yield f"\n\n[stream error: {e}]"

    # ============================ public api ============================
    def query(
        self,
        question: str,
        document_id: Optional[int] = None,
        top_k: int = 5,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        chunks, diag = self.retrieve(question, document_id, top_k, history or [])
        answer = self.generate(question, chunks, history or [])
        return {
            "answer": answer,
            "sources": self._sources(chunks),
            "diagnostics": diag,
        }

    def query_stream(
        self,
        question: str,
        document_id: Optional[int] = None,
        top_k: int = 5,
        history: Optional[List[Dict]] = None,
    ) -> Tuple[Iterable[str], List[Dict], Dict]:
        chunks, diag = self.retrieve(question, document_id, top_k, history or [])
        return self.generate_stream(question, chunks, history or []), self._sources(chunks), diag

    @staticmethod
    def _sources(chunks: List[Dict]) -> List[Dict]:
        out = []
        for c in chunks:
            text = c["chunk_text"]
            out.append({
                "text": text[:300] + ("..." if len(text) > 300 else ""),
                "similarity": float(c.get("similarity") or 0.0),
                "rrf_score": float(c.get("rrf_score") or 0.0),
                "rerank_score": float(c.get("rerank_score") or 0.0),
                "filename": c["filename"],
                "section": c.get("parent_section") or "",
                "chunk_index": c["chunk_index"],
            })
        return out


rag_service = RAGService()
