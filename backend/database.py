"""
Postgres + pgvector access layer.

  - True batch inserts via execute_values
  - Hybrid retrieval: dense (HNSW cosine) + sparse (BM25 via tsvector / ts_rank)
    fused with Reciprocal Rank Fusion in pure SQL.
  - Sentence-window expansion: fetch chunk + its neighbours (chunk_index ±1)
  - Chat-session storage for multi-turn memory.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List, Dict, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:2060@localhost:5433/rag_database"
)


class Database:
    def __init__(self) -> None:
        self.connection_params = self._parse_db_url(DATABASE_URL)

    @staticmethod
    def _parse_db_url(url: str) -> Dict[str, str]:
        url = url.replace("postgresql://", "")
        auth, location = url.split("@")
        user, password = auth.split(":", 1)
        host_port, dbname = location.split("/", 1)
        host, port = host_port.split(":")
        return {"host": host, "port": port, "database": dbname, "user": user, "password": password}

    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --------------------------- documents ---------------------------
    def insert_document(self, filename: str, file_type: str, file_data: bytes) -> int:
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, file_type, file_data, file_size)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (filename, file_type, psycopg2.Binary(file_data), len(file_data)),
            )
            return cur.fetchone()[0]

    def insert_chunks_batch(
        self,
        document_id: int,
        rows: Sequence[Tuple[str, int, list, Optional[str], Optional[str], str]],
    ) -> None:
        """
        rows: (chunk_text, chunk_index, embedding_list, parent_section,
               context_summary, chunk_type)
        """
        if not rows:
            return
        with self.get_connection() as conn, conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO document_chunks
                    (document_id, chunk_text, chunk_index, embedding,
                     parent_section, context_summary, chunk_type)
                VALUES %s
                """,
                [
                    (
                        document_id,
                        chunk_text,
                        chunk_index,
                        embedding,
                        parent_section,
                        context_summary,
                        chunk_type,
                    )
                    for (chunk_text, chunk_index, embedding,
                         parent_section, context_summary, chunk_type) in rows
                ],
                template="(%s, %s, %s, %s::vector, %s, %s, %s)",
                page_size=200,
            )

    # --------------------------- retrieval ---------------------------
    def hybrid_search(
        self,
        query_embedding,
        query_text: str,
        top_k: int = 20,
        document_id: Optional[int] = None,
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion of dense (HNSW cosine) + sparse (BM25 / tsvector)
        results, computed in a single SQL query.
        """
        emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        candidate_limit = top_k * 4

        if document_id:
            doc_where_vec = "WHERE dc.document_id = %s"
            doc_where_bm = "AND dc.document_id = %s"
            params = [
                emb, emb, document_id, emb, candidate_limit,
                query_text, query_text, query_text, document_id, candidate_limit,
                top_k,
            ]
        else:
            doc_where_vec = ""
            doc_where_bm = ""
            params = [
                emb, emb, emb, candidate_limit,
                query_text, query_text, query_text, candidate_limit,
                top_k,
            ]

        sql = f"""
        WITH vec AS (
            SELECT
                dc.id,
                ROW_NUMBER() OVER (ORDER BY dc.embedding <=> %s::vector) AS rank,
                1 - (dc.embedding <=> %s::vector) AS sim
            FROM document_chunks dc
            {doc_where_vec}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        ),
        bm AS (
            SELECT
                dc.id,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(dc.tsv, plainto_tsquery('english', %s)) DESC
                ) AS rank,
                ts_rank_cd(dc.tsv, plainto_tsquery('english', %s)) AS bm25
            FROM document_chunks dc
            WHERE dc.tsv @@ plainto_tsquery('english', %s)
            {doc_where_bm}
            ORDER BY bm25 DESC
            LIMIT %s
        ),
        fused AS (
            SELECT
                COALESCE(vec.id, bm.id) AS id,
                COALESCE(1.0 / ({rrf_k} + vec.rank), 0) +
                COALESCE(1.0 / ({rrf_k} + bm.rank), 0) AS rrf_score,
                vec.sim AS dense_sim,
                bm.bm25 AS bm25_score
            FROM vec
            FULL OUTER JOIN bm ON vec.id = bm.id
        )
        SELECT
            dc.id, dc.chunk_text, dc.chunk_index, dc.parent_section,
            dc.context_summary, dc.chunk_type, dc.document_id, d.filename,
            f.rrf_score,
            COALESCE(f.dense_sim, 0) AS similarity,
            COALESCE(f.bm25_score, 0) AS bm25_score
        FROM fused f
        JOIN document_chunks dc ON dc.id = f.id
        JOIN documents d ON dc.document_id = d.id
        ORDER BY f.rrf_score DESC
        LIMIT %s
        """

        with self.get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def get_neighbors(self, document_id: int, chunk_index: int, window: int = 1) -> List[Dict]:
        """Return chunks at index-window..index+window from the same document."""
        with self.get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, chunk_text, chunk_index, parent_section
                FROM document_chunks
                WHERE document_id = %s
                  AND chunk_index BETWEEN %s AND %s
                ORDER BY chunk_index
                """,
                (document_id, chunk_index - window, chunk_index + window),
            )
            return list(cur.fetchall())

    # --------------------------- documents meta ---------------------------
    def get_latest_document_id(self) -> Optional[int]:
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM documents ORDER BY uploaded_at DESC LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None

    def get_document_info(self, document_id: int) -> Optional[Dict]:
        with self.get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, filename, file_type, file_size, uploaded_at FROM documents WHERE id = %s",
                (document_id,),
            )
            return cur.fetchone()

    def get_document_file(self, document_id: int) -> Optional[Dict]:
        with self.get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT filename, file_type, file_data FROM documents WHERE id = %s",
                (document_id,),
            )
            return cur.fetchone()

    def get_all_documents(self) -> List[Dict]:
        with self.get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, filename, file_type, file_size, uploaded_at "
                "FROM documents ORDER BY uploaded_at DESC"
            )
            return list(cur.fetchall())

    def delete_document(self, document_id: int) -> None:
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))

    def health_check(self) -> bool:
        try:
            with self.get_connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception as e:
            print(f"[db] health check failed: {e}")
            return False

    # --------------------------- chat memory ---------------------------
    def ensure_session(self, session_id: str) -> None:
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (session_id) VALUES (%s) "
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id,),
            )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self.ensure_session(session_id)
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            return list(reversed(cur.fetchall()))


db = Database()
