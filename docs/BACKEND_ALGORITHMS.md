# Backend Algorithms — What, How, and Why

This document explains every algorithm and model used in the backend, what it does, why it was chosen over alternatives, and where it lives in the code.

---

## 1. Text Extraction (PDF Multi-Method Fallback)

**Where:** [backend/document_service.py](../backend/document_service.py) — `_extract_pdf`

**What it does:** Tries three extractors in order and stops at the first that produces > 100 chars of meaningful text.

| Order | Method | Strength | Weakness |
|---|---|---|---|
| 1 | **PyPDF2** | Fast, pure Python | Bad with complex layouts, no OCR |
| 2 | **pdfplumber** | Handles tables, multi-column | Slower; still fails on scanned PDFs |
| 3 | **Tesseract OCR** (via `pdf2image`) | Reads scanned/image PDFs | Slow (~5 s/page at 300 DPI), needs `poppler` + `tesseract` binaries |

**Why a fallback chain instead of one method?**
PDFs are wildly heterogeneous. A single extractor fails ~20–30% of the time on real-world inputs. The cascade gives near-100% success without paying OCR cost on text-based PDFs.

---

## 2. Chunking — Hierarchical (for prose / docs)

**Where:** [backend/chunkers/hierarchical.py](../backend/chunkers/hierarchical.py)

**What it does:**
1. Detects headings via four regex patterns:
   - Markdown `# / ##`
   - Numbered headings (`1.2.3 Title`)
   - PDF `--- Page N ---` markers
   - ALL-CAPS lines (≤ 12 words) — fallback for unformatted PDFs
2. Builds a section tree where each chunk's `parent_section` is the full path (`H1 > H2 > H3`).
3. Within each section, splits with **`RecursiveCharacterTextSplitter`** (LangChain) using separator priority `["\n\n", "\n", ". ", " ", ""]` so it breaks on paragraphs first, sentences second, words last.

**Parameters:** `chunk_size=800`, `chunk_overlap=120` (~15% overlap so concepts spanning a chunk boundary aren't lost).

**Why hierarchical over fixed-size?**
Fixed-size chunking loses two things: (1) section context, and (2) it routinely splits a sentence or even a single thought across two chunks. Hierarchical chunking respects document structure, and feeding the section path into the embedding (see §6) gives retrieval a huge precision boost on multi-topic documents.

**Why `RecursiveCharacterTextSplitter` over `CharacterTextSplitter`?**
`Recursive` tries the highest-quality boundary first and only descends to character-level cuts if forced. Single `CharacterTextSplitter` would cut mid-word.

**Alternatives considered:**
- `SemanticChunker` (embeds adjacent sentences, splits on similarity drop): higher quality but ~20× slower at ingest. Not worth it for the average user.
- Fixed token windows: simpler but loses structure.

---

## 3. Chunking — AST (for source code)

**Where:** [backend/chunkers/code_ast.py](../backend/chunkers/code_ast.py)

**What it does:**
- **Python (`.py`)**: uses the built-in **`ast`** module to walk the syntax tree. Each top-level function, class, or method becomes a chunk. Class bodies are descended so methods are chunked individually with qualified name `ClassA.method_b`. Module-level imports are gathered into one synthetic chunk.
- **Other languages** (`.js .ts .tsx .go .java .rs .cpp .c .rb .php .cs .kt .swift`): uses **`tree-sitter`** via `tree_sitter_language_pack` (pre-built grammars). Walks for `function_declaration`, `class_declaration`, `method_definition`, `interface_declaration`, etc.
- **Fallback**: if a grammar isn't available or parsing fails, drops to recursive text splitting on natural code boundaries (`\nclass `, `\ndef `, `\nfunction `).

**Why AST over text splitting for code?**
Text splitters cut functions mid-body. The retriever then returns half a function and the LLM hallucinates the rest. AST chunking guarantees each chunk is a complete syntactic unit, and the qualified symbol name (`ClassA.method_b`) becomes free metadata for filtering and display.

**Why `ast` for Python instead of tree-sitter?**
`ast` is in the stdlib (no compile/install), faster, and bundled with the runtime — for Python it's strictly better.

**Why tree-sitter for everything else?**
It's the same parser library used by GitHub, Atom, and Neovim. One library covers ~30 languages with mature grammars. Alternatives would require shipping per-language ANTLR/PLY parsers.

---

## 4. Embedding Model — `BAAI/bge-small-en-v1.5`

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `RAGService.__init__`, `embed`

**What it does:** Maps text → 384-dim L2-normalised vector. Same dimensionality as the previous model (`all-MiniLM-L6-v2`) so the DB schema didn't change, but state-of-the-art quality.

**Benchmarks (MTEB English, v1.5 vs MiniLM-L6-v2):**
| Task | MiniLM-L6 | bge-small-en-v1.5 |
|---|---|---|
| Avg retrieval | 41.9 | **51.7** |
| Avg classification | 63.1 | **74.1** |
| Avg overall | 56.3 | **62.2** |

**Why `bge-small` instead of `bge-base` or `bge-large`?**
- `small` (33 M params, 130 MB): runs on CPU at ~2 ms/query.
- `base` (~110 M, 440 MB): ~3 points better but 4× slower.
- `large` (~335 M, 1.3 GB): ~5 points better, 12× slower, can't fit in this project's footprint comfortably.
- For a single-user/small-team app, `small` is the sweet spot.

**Why normalise (`normalize_embeddings=True`)?**
With L2-normalised vectors, cosine similarity == inner product, and pgvector's `<=>` operator works directly on cosine distance. No re-normalisation in SQL.

---

## 5. Reranker — `BAAI/bge-reranker-base` (Cross-Encoder)

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `retrieve` (rerank step)

**What it does:** Takes the top ~20 candidates from hybrid search, scores `(query, chunk)` pairs jointly with a cross-encoder, sorts by that score, returns the top `top_k` (default 5).

**Bi-encoder vs Cross-encoder:**
| Property | Bi-encoder (BGE-small) | Cross-encoder (BGE-reranker) |
|---|---|---|
| Encoding | query and chunk separately | query and chunk together (full attention) |
| Cost | O(N) at index, O(1) per query | O(N) per query — much more expensive |
| Quality | Coarse | Sharper by ~10–15 nDCG points |
| Use | First-pass retrieval | Refine top-N |

**Why two stages?**
Cross-encoder on the full corpus would be O(corpus_size) per query — unaffordable. The classic pattern: **fast bi-encoder retrieval → slow cross-encoder rerank on a shortlist**. Best of both.

**Why `bge-reranker-base` over `bge-reranker-large` or `cohere-rerank`?**
- `base` (~110 M, ~440 MB): ~50 ms for 20 pairs on CPU.
- `large`: better, but ~3× slower.
- Cohere/Voyage rerankers: better but cost money + add network latency.
- Disable with `USE_RERANKER=0` if you need pure speed.

---

## 6. Contextual Retrieval (Anthropic 2024 technique)

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `embed_and_store_chunks` → `_chunk_context`

**What it does:** Before embedding each chunk, calls Gemini Flash with the full document + the chunk and asks for a one-sentence "where this chunk fits" summary. This summary is prepended to the chunk text before embedding (and is also stored in the `context_summary` column for BM25 scoring).

**Embed input layout:**
```
Section: <parent_section>
Context: <generated one-line summary>
<chunk_text>
```

**Why this matters:**
A bare chunk like "It returns 0 if invalid." is meaningless. With `Context: This excerpt explains the return value of validate_email().`, the embedding now correctly clusters with queries like "what does validate_email return". Anthropic reported a **49% reduction in retrieval failures** with this technique alone.

**Cost:** One Gemini Flash call per chunk at ingest time (cheap, batched, fire-and-forget). Zero cost at query time.

**Disable with** `USE_CONTEXTUAL_RETRIEVAL=0` if you have very large documents and want fast ingest.

---

## 7. Query Rewriting

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `_rewrite_query`

**What it does:** Rewrites the user's question into a self-contained search query before any retrieval. Does three jobs:
1. Expands acronyms (`ml` → `machine learning`)
2. Resolves pronouns from chat history (`how does it scale?` → `how does machine learning scale?`)
3. Adds 1–2 synonyms for recall

**Triggered when:** the question is short (≤ 4 words) **or** there's prior chat history.

**Why?**
Conversational follow-ups are the #1 reason RAG systems fail in production — embeddings don't know what "it" or "that" means. Rewriting normalises every query to a self-contained form.

---

## 8. HyDE — Hypothetical Document Embeddings

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `_hyde`

**What it does:** For short queries (≤ 4 words), instead of embedding the query directly, asks Gemini to write a 2–3 sentence plausible *answer* and embeds that.

**Why?**
A query like "what is ml?" produces a weak, generic embedding. The synthetic passage `"Machine learning is a branch of AI in which algorithms learn patterns from data..."` produces an embedding that lives much closer to actual document chunks about ML.

**Tradeoff:** adds one Gemini call (~300 ms). Only triggered for short queries where the win is largest.

**Source:** Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels".

---

## 9. Hybrid Search (Dense + BM25, fused with RRF)

**Where:** [backend/database.py](../backend/database.py) — `hybrid_search`

**What it does:** A single SQL query with three CTEs:

```sql
WITH vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $emb) AS rank, ...
    FROM document_chunks
    LIMIT $candidate_limit
),
bm AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery($q)) DESC) AS rank, ...
    FROM document_chunks
    WHERE tsv @@ plainto_tsquery($q)
    LIMIT $candidate_limit
),
fused AS (
    SELECT COALESCE(vec.id, bm.id) AS id,
           1.0 / (60 + vec.rank) + 1.0 / (60 + bm.rank) AS rrf_score
    FROM vec FULL OUTER JOIN bm ON vec.id = bm.id
)
SELECT ... ORDER BY fused.rrf_score DESC LIMIT $top_k
```

### 9a. Dense retrieval — pgvector + HNSW

- Cosine distance via the `<=>` operator on the `embedding vector(384)` column.
- Indexed with **HNSW** (m=16, ef_construction=64).

**Why HNSW over IVFFlat?**
| | IVFFlat | HNSW |
|---|---|---|
| Recall | 80–90% with default `lists` | 95–99% out of the box |
| Tuning | Needs `lists` ≈ √N tuned per dataset | Self-tuning |
| Build time | Fast | Slower (~3×) |
| Query time | Comparable | Comparable |
| Memory | Lower | Higher |

For < 10 M chunks the HNSW recall win dominates. The original schema used IVFFlat — replaced.

### 9b. Sparse retrieval — Postgres BM25 (`tsvector` + `ts_rank_cd`)

- Auto-maintained `tsv` column via trigger:
  ```sql
  setweight(to_tsvector('english', parent_section), 'A') ||
  setweight(to_tsvector('english', context_summary), 'B') ||
  setweight(to_tsvector('english', chunk_text), 'C')
  ```
- GIN index for sub-millisecond keyword lookup.
- `ts_rank_cd` is "cover density" ranking — closer to BM25 than plain `ts_rank`.

**Why give section/context higher weight?**
A keyword in a heading is a much stronger signal than the same word buried in body text.

### 9c. Fusion — Reciprocal Rank Fusion (RRF)

```
score(d) = Σ_methods 1 / (k + rank_method(d))     k = 60
```

**Why RRF over weighted score combination?**
Vector similarity (cosine, range −1..1) and BM25 score (unbounded float) live on incomparable scales. Trying to combine them as `α·sim + β·bm25` requires per-dataset tuning. RRF only uses **rank**, so it's scale-free, parameter-light, and Microsoft / Pinecone / Weaviate all default to it. The constant `k=60` is the empirical sweet spot from the original Cormack 2009 paper.

**Why FULL OUTER JOIN?**
A chunk that's only retrieved by BM25 (e.g. exact ID match) or only by the vector index (semantic match) still gets a (lower) RRF score. Important for queries where one method completely fails — like the "what is ml?" case where BM25 finds the literal "ML" but the vector embedding is too generic.

---

## 10. Sentence-Window Retrieval

**Where:** [backend/database.py](../backend/database.py) — `get_neighbors`; [backend/rag_service.py](../backend/rag_service.py) — `retrieve` (window step)

**What it does:** Each retrieved chunk is enriched with `chunk_index ± 1` from the same document. The LLM sees the **window** as context, not just the matching chunk.

**Why?**
The matching chunk is *sharp* (high precision for retrieval), but the answer often needs information from the surrounding paragraphs. The window strategy gives the best of both: precise matching + broad context.

**Source:** LlamaIndex sentence-window pattern.

---

## 11. Citation-Grounded Generation

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `_build_prompt`

The prompt:
1. Numbers each context block `[1]`, `[2]`, ...
2. Includes the section path and source filename in each block header
3. Includes the last 6 turns of chat history for follow-up resolution
4. Instructs the LLM:
   - Cite inline as `[1]`, `[2]`
   - Use ONLY the provided context
   - Say "the context does not contain this" if not supported

**Why?**
- Citations make hallucinations **visible** — an unsupported claim has no `[N]` next to it.
- The frontend ([templates/app.html](../templates/app.html)) parses `[N]` and renders them as clickable badges that scroll to the source card.

---

## 12. Streaming (SSE)

**Where:** [backend/main.py](../backend/main.py) — `chat_stream`; [backend/rag_service.py](../backend/rag_service.py) — `generate_stream`

**What it does:** `POST /chat/stream` returns Server-Sent Events:
```
data: {"type":"meta","sources":[...],"diagnostics":{...}}
data: {"type":"token","text":"Machine "}
data: {"type":"token","text":"learning "}
...
data: {"type":"done"}
```

The frontend renders sources immediately (so the user sees what was retrieved while the LLM thinks), then types the answer character-by-character.

**Why SSE over WebSockets?**
- Server-to-client only is sufficient — no need for bidirectional channel.
- Plain HTTP — works through any proxy, no upgrade handshake.
- Native `EventSource` in browsers, trivial parsing.

---

## 13. Chat Memory (Multi-Turn)

**Where:** Tables `chat_sessions` + `chat_messages` in [init.sql](../init.sql); [backend/database.py](../backend/database.py) — `get_history`, `append_message`; [backend/main.py](../backend/main.py) — `/chat`, `/chat/stream`

**What it does:** Frontend persists a `session_id` (UUID) in `localStorage`. Every chat request sends it. Backend appends user + assistant messages and loads the last 10 turns into the prompt so follow-ups work ("explain that more", "and how does it compare to X").

**Why server-side history vs client-side only?**
Survives page reloads and lets the rewrite step on the backend see the full conversation without trusting the client.

---

## 14. Embedding Cache (Query-Side)

**Where:** [backend/rag_service.py](../backend/rag_service.py) — `_cached_query_embed` (`functools.lru_cache(maxsize=512)`)

**What it does:** Identical query strings reuse the cached embedding instead of re-running the model.

**Why?**
Repeated queries (refresh, "ask again") are common in chat. ~2 ms saved per cache hit, near-zero cost.

---

## 15. Batch Insert (`execute_values`)

**Where:** [backend/database.py](../backend/database.py) — `insert_chunks_batch`

**What it does:** Single round-trip multi-row `INSERT` via `psycopg2.extras.execute_values` with `page_size=200`.

**Why?**
The original code looped one `INSERT` per chunk (1000 round-trips per 1000-chunk doc → seconds of pure latency). `execute_values` is ~20–50× faster and is the canonical way to bulk-load via psycopg2.

---

## Pipeline Latency Budget (typical query, CPU)

| Step | Time |
|---|---|
| Query rewrite (Gemini) | ~300 ms (only when triggered) |
| HyDE (Gemini) | ~300 ms (only for short queries) |
| Embed query | ~5 ms (cached: 0) |
| Hybrid search SQL | ~10–30 ms |
| Cross-encoder rerank (20 pairs) | ~60 ms |
| Sentence-window fetch | ~5 ms |
| First Gemini token | ~400–700 ms |
| **Total time-to-first-token** | **~0.8–1.5 s** |
| Subsequent tokens | streamed at Gemini's rate (~50/sec) |

---

## Tunable Knobs (Summary)

| Knob | Where | Default | Notes |
|---|---|---|---|
| `top_k` | request body / UI slider | 5 | After rerank; pre-rerank candidates = 4× this |
| `chunk_size` | `HierarchicalChunker(...)` | 800 chars | Smaller = more precise, more chunks |
| `chunk_overlap` | same | 120 chars | ~15% — keep concepts intact across boundaries |
| `rrf_k` | `db.hybrid_search(rrf_k=60)` | 60 | Cormack-paper sweet spot |
| HNSW `m`, `ef_construction` | [init.sql](../init.sql) | 16, 64 | Recall vs build time |
| `USE_RERANKER` | env | 1 | Disable for max speed |
| `USE_CONTEXTUAL_RETRIEVAL` | env | 1 | Disable for fast ingest |
| Reranker candidate count | `top_k * 4` | 20 | More candidates = better rerank, slower |
| Sentence-window size | `db.get_neighbors(window=1)` | 1 | Increase for more context |
| History turns sent to LLM | `_build_prompt` | 6 | More = better follow-ups, more tokens |
