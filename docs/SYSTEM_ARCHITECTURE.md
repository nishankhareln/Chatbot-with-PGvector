# System Architecture

## Overview

A document-grounded chatbot built on the **RAG** (Retrieval-Augmented Generation) pattern, with a stack of advanced retrieval techniques layered on top of a basic dense-vector search.

```
┌──────────────┐        ┌────────────────────────┐        ┌───────────────────┐
│   Browser    │ <───>  │   FastAPI Backend      │ <───>  │  Postgres +       │
│  (HTML / SSE)│        │  (uvicorn, port 8000)  │        │  pgvector (5433)  │
└──────────────┘        └──────────┬─────────────┘        └───────────────────┘
                                   │
                                   ├── SentenceTransformer (BGE-small)
                                   ├── CrossEncoder (BGE-reranker)
                                   └── Google Gemini (Flash)
```

---

## Components

### 1. Frontend
| File | Role |
|---|---|
| [templates/app.html](../templates/app.html) | Single-page app — Tailwind UI, SSE streaming client, citation rendering, drag-and-drop uploads. Served at `GET /`. |
| [frontend/app.py](../frontend/app.py) | Optional Streamlit alternative on port 8501. |

### 2. FastAPI Backend ([backend/main.py](../backend/main.py))

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves `templates/app.html` |
| `GET /health` | Backend + db + reranker status |
| `POST /upload` | Multi-file ingest (PDF, MD, TXT, code) |
| `POST /chat` | Synchronous Q&A (returns full answer + sources) |
| `POST /chat/stream` | Server-Sent Events stream of tokens + meta |
| `GET /documents`, `/document/{id}`, `/document/{id}/download`, `DELETE /document/{id}` | Document management |

### 3. Document Service ([backend/document_service.py](../backend/document_service.py))

Extracts text + dispatches to the right chunker.

| File type | Extractor | Chunker |
|---|---|---|
| `.pdf` | PyPDF2 → pdfplumber → OCR (fallback chain) | `HierarchicalChunker` |
| `.md`, `.markdown`, `.txt` | UTF-8 read | `HierarchicalChunker` |
| `.py` | Built-in `ast` | `CodeASTChunker` (functions, classes, methods) |
| `.js`, `.ts`, `.go`, `.java`, `.rs`, `.cpp`, `.c`, `.rb`, `.php`, `.cs`, `.kt`, `.swift` | Tree-sitter via `tree_sitter_language_pack` | `CodeASTChunker` |

### 4. Chunkers ([backend/chunkers/](../backend/chunkers/))

| Class | Strategy |
|---|---|
| `HierarchicalChunker` | Detects markdown headers (`#`, `##`), numbered headings (`1.2 Title`), `--- Page N ---` markers, ALL-CAPS lines → builds a section tree → splits each section with `RecursiveCharacterTextSplitter`, attaching the full path (`H1 > H2 > H3`) as `parent_section`. |
| `CodeASTChunker` | Splits by syntactic nodes. Each chunk records its qualified symbol (`ClassA.method_b`). Falls back to text chunking if grammar isn't available. |

### 5. RAG Service ([backend/rag_service.py](../backend/rag_service.py))

The retrieval + generation pipeline.

#### Ingest pipeline
1. Document is chunked (above).
2. **Contextual retrieval**: Gemini Flash writes a one-sentence "where this fits" summary per chunk (`Context: ...`).
3. Embedding input = `Section: <path>\nContext: <summary>\n<chunk_text>`.
4. Embedded with **`BAAI/bge-small-en-v1.5`** (384-dim).
5. Stored in `document_chunks` with vector + parent_section + context_summary + chunk_type. A trigger auto-builds the `tsv` (BM25) column.

#### Query pipeline
```
user question
   ↓
[1] Query rewriting          — Gemini expands acronyms, resolves pronouns from history
   ↓
[2] HyDE (short queries)     — Gemini drafts a hypothetical answer; we embed THAT
   ↓
[3] Hybrid retrieval         — Postgres CTE:
                                  · vec  = HNSW cosine top-N
                                  · bm   = ts_rank_cd top-N
                               fused via Reciprocal Rank Fusion in SQL
   ↓
[4] Cross-encoder rerank     — BGE-reranker-base scores (query, chunk) pairs
   ↓
[5] Sentence-window expand   — fetch chunk_index ±1 from same document
   ↓
[6] Prompt build             — context blocks numbered [1], [2], ...; chat history
                               appended; LLM instructed to cite inline
   ↓
[7] Stream from Gemini       — SSE tokens to frontend; final answer cached in chat memory
```

### 6. Database ([backend/database.py](../backend/database.py), [init.sql](../init.sql))

**Postgres 16 + pgvector**.

| Table | Columns |
|---|---|
| `documents` | id, filename, file_type, file_data (BYTEA), file_size, uploaded_at |
| `document_chunks` | id, document_id, chunk_text, chunk_index, **embedding vector(384)**, **parent_section**, **context_summary**, **chunk_type**, **tsv tsvector**, created_at |
| `chat_sessions` | session_id (UUID), created_at |
| `chat_messages` | session_id, role, content, created_at |

**Indexes:**
| Name | Type | Use |
|---|---|---|
| `document_chunks_embedding_hnsw_idx` | HNSW (m=16, ef_construction=64) | Dense vector search |
| `document_chunks_tsv_idx` | GIN over tsvector | BM25 keyword search |
| `document_chunks_text_trgm_idx` | GIN trigram | Fuzzy fallback |
| `document_chunks_doc_chunk_idx` | btree (document_id, chunk_index) | Sentence-window fetch |

**Trigger:**
- `document_chunks_tsv_update` — auto-maintains `tsv` from `parent_section` (weight A) + `context_summary` (B) + `chunk_text` (C).

---

## Data Flow — End to End

### Upload
```
Browser → POST /upload
       → DocumentService.process_document(path, ext)
            → extract text (PDF chain / text read)
            → HierarchicalChunker OR CodeASTChunker
       → db.insert_document(...)
       → RAGService.embed_and_store_chunks(doc_id, chunks)
            → for each chunk: Gemini → context summary
            → BGE-small batch embed
            → db.insert_chunks_batch(...)   [execute_values]
                  ↓ trigger fires → tsv column populated
       → response: chunk count per file
```

### Chat (streaming)
```
Browser → POST /chat/stream  { question, top_k, session_id }
       → load history from chat_messages
       → RAGService.query_stream(...)
            → query rewrite (Gemini)
            → HyDE if short (Gemini)
            → db.hybrid_search(...)         [vec + bm25 + RRF in one SQL]
            → reranker.predict(pairs)       [BGE cross-encoder]
            → db.get_neighbors(...)         [sentence window]
            → build prompt with [1] [2] context blocks
            → client.models.generate_content_stream(...)
       → SSE event: "meta" { sources, diagnostics }
       → SSE events: "token" { text }       (multiple)
       → SSE event:  "done"
       → append assistant message to chat_messages
Browser:
       → render diagnostics row
       → render sources panel
       → typewrite tokens with cursor
       → replace [1] [2] in answer with clickable badges
```

---

## Models & External Services

| Component | Model | Where it runs |
|---|---|---|
| Embedder | `BAAI/bge-small-en-v1.5` (384 dim) | In-process, CPU |
| Reranker | `BAAI/bge-reranker-base` | In-process, CPU |
| LLM (rewrite, HyDE, context, generation) | `gemini-2.5-flash` | Google Gemini API |
| Vector DB | pgvector 16 | Docker container |
| OCR (PDF fallback) | Tesseract 5.x | OS package in image |

---

## Configuration (env vars)

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Required for generation |
| `DATABASE_URL` | `postgresql://postgres:2060@localhost:5433/rag_database` | DB connection |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Override embedder |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Override reranker |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `USE_RERANKER` | `1` | Set `0` to skip rerank stage |
| `USE_CONTEXTUAL_RETRIEVAL` | `1` | Set `0` to skip per-chunk Gemini context calls |
| `TESSERACT_CMD` | auto-detect | Path to tesseract binary if non-standard |

---

## Why each technique?

| Technique | Problem it solves |
|---|---|
| Hierarchical chunking | Naïve fixed-size chunks split sentences and lose section context |
| AST chunking | Splitting code on character boundaries breaks functions mid-body |
| Contextual retrieval | A bare chunk like "He returns 0 if invalid" is useless without knowing it's about `validate_email`. Adding a one-line context fixes it. |
| Hybrid search (vector + BM25) | Pure dense search misses literal token matches (acronyms, IDs); pure BM25 misses semantic matches. RRF gets both. |
| HyDE | Short queries ("what is ml?") have weak embeddings. A hypothetical full answer has a much richer embedding. |
| Cross-encoder rerank | Bi-encoder retrieval is fast but coarse. Cross-encoder reads (query, chunk) jointly and orders much more accurately — applied only to the top ~20. |
| Sentence-window | The matching chunk is sharp, but its neighbours give the LLM enough context to answer. |
| HNSW index | Better recall than IVFFlat with no `lists` tuning. |
| Streaming + chat memory | Time-to-first-token feels instant; multi-turn questions resolve "it" / "that" correctly. |
| Citation enforcement | Forces the LLM to ground every claim, makes hallucinations visible. |

---

## File Layout

```
SimpleChatbot/
├── backend/
│   ├── chunkers/
│   │   ├── __init__.py
│   │   ├── base.py             # Chunk dataclass
│   │   ├── hierarchical.py     # Section-tree chunker
│   │   └── code_ast.py         # ast + tree-sitter
│   ├── database.py             # pgvector + hybrid SQL
│   ├── document_service.py     # extract + dispatch
│   ├── main.py                 # FastAPI routes + SSE
│   └── rag_service.py          # full pipeline
├── frontend/
│   └── app.py                  # Streamlit (optional)
├── templates/
│   └── app.html                # Main HTML UI (served at /)
├── docs/
│   ├── HOW_TO_RUN.md
│   └── SYSTEM_ARCHITECTURE.md  ← you are here
├── init.sql                    # Schema + indexes + trigger
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── check_database.py
└── .env
```
