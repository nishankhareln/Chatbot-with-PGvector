-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_data BYTEA NOT NULL,
    file_size INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table with vector + BM25 + hierarchical metadata + contextual summary
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(384),
    parent_section TEXT,
    context_summary TEXT,
    chunk_type VARCHAR(32) DEFAULT 'text',
    tsv tsvector,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backfill columns if upgrading an existing DB
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_section TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS context_summary TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(32) DEFAULT 'text';
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS tsv tsvector;

-- Auto-maintained tsvector for BM25 / keyword search
CREATE OR REPLACE FUNCTION document_chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv :=
        setweight(to_tsvector('english', coalesce(NEW.parent_section, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.context_summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.chunk_text, '')), 'C');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS document_chunks_tsv_update ON document_chunks;
CREATE TRIGGER document_chunks_tsv_update
BEFORE INSERT OR UPDATE ON document_chunks
FOR EACH ROW EXECUTE FUNCTION document_chunks_tsv_trigger();

-- Drop old ivfflat index if present, replace with HNSW (better recall, no list tuning)
DROP INDEX IF EXISTS document_chunks_embedding_idx;
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- BM25 index
CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
ON document_chunks USING GIN (tsv);

-- Trigram index for fuzzy filename / fallback search
CREATE INDEX IF NOT EXISTS document_chunks_text_trgm_idx
ON document_chunks USING GIN (chunk_text gin_trgm_ops);

-- Lookup helpers
CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
ON document_chunks(document_id);

CREATE INDEX IF NOT EXISTS document_chunks_doc_chunk_idx
ON document_chunks(document_id, chunk_index);

-- Chat sessions for multi-turn memory
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx
ON chat_messages(session_id, created_at);
