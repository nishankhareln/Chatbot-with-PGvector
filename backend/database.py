import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:2060@localhost:5433/rag_database")


class Database:
    def __init__(self):
        self.connection_params = self._parse_db_url(DATABASE_URL)
    
    def _parse_db_url(self, url):
        """Parse DATABASE_URL into connection parameters"""
        # Remove postgresql:// prefix
        url = url.replace("postgresql://", "")
        
        # Split into user:pass@host:port/dbname
        auth, location = url.split("@")
        user, password = auth.split(":")
        host_port, dbname = location.split("/")
        host, port = host_port.split(":")
        
        return {
            "host": host,
            "port": port,
            "database": dbname,
            "user": user,
            "password": password
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def insert_document(self, filename, file_type, file_data):
        """
        Insert document metadata and binary data, return document_id
        
        Args:
            filename: Name of the file
            file_type: Extension (pdf, txt, md)
            file_data: Binary file content (bytes)
            
        Returns:
            document_id (int)
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (filename, file_type, file_data, file_size) 
                    VALUES (%s, %s, %s, %s) RETURNING id
                    """,
                    (filename, file_type, psycopg2.Binary(file_data), len(file_data))
                )
                document_id = cur.fetchone()[0]
                return document_id
    
    def insert_chunk(self, document_id, chunk_text, chunk_index, embedding):
        """Insert a document chunk with its embedding"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, chunk_text, chunk_index, embedding.tolist())
                )
    
    def insert_chunks_batch(self, document_id, chunks_data):
        """Insert multiple chunks at once for better performance"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for chunk_text, chunk_index, embedding in chunks_data:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (document_id, chunk_text, chunk_index, embedding.tolist())
                    )
    
    def similarity_search(self, query_embedding, top_k=3, document_id=None):
        """Find most similar chunks using cosine similarity"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if document_id:
                    # Search within specific document
                    cur.execute(
                        """
                        SELECT 
                            dc.chunk_text,
                            dc.chunk_index,
                            d.filename,
                            1 - (dc.embedding <=> %s::vector) as similarity
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.document_id = %s
                        ORDER BY dc.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding.tolist(), document_id, query_embedding.tolist(), top_k)
                    )
                else:
                    # Search across all documents
                    cur.execute(
                        """
                        SELECT 
                            dc.chunk_text,
                            dc.chunk_index,
                            d.filename,
                            1 - (dc.embedding <=> %s::vector) as similarity
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        ORDER BY dc.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding.tolist(), query_embedding.tolist(), top_k)
                    )
                
                results = cur.fetchall()
                return results
    
    def get_latest_document_id(self):
        """Get the most recently uploaded document ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM documents ORDER BY uploaded_at DESC LIMIT 1")
                result = cur.fetchone()
                return result[0] if result else None
    
    def get_document_info(self, document_id):
        """Get document metadata (without binary data)"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, filename, file_type, file_size, uploaded_at FROM documents WHERE id = %s",
                    (document_id,)
                )
                return cur.fetchone()
    
    def get_document_file(self, document_id):
        """Get document binary data for download"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT filename, file_type, file_data FROM documents WHERE id = %s",
                    (document_id,)
                )
                return cur.fetchone()
    
    def get_all_documents(self):
        """Get all documents metadata (without binary data)"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, filename, file_type, file_size, uploaded_at FROM documents ORDER BY uploaded_at DESC"
                )
                return cur.fetchall()
    
    def delete_document(self, document_id):
        """Delete document and all its chunks (cascade)"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    
    def health_check(self):
        """Check if database connection is healthy"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            print(f"Database health check failed: {e}")
            return False


# Singleton instance
db = Database()