from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict
from database import db

load_dotenv()

# Configure Gemini with new API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RAGService:
    def __init__(self):
        """Initialize RAG service with embedding model and LLM"""
        # Load free embedding model (384 dimensions)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Gemini client (new API)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        print("RAG Service initialized")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (numpy array)
        """
        embedding = self.embedding_model.encode(text)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def embed_and_store_chunks(self, document_id: int, chunks: List[str]):
        """
        Generate embeddings for chunks and store in database
        
        Args:
            document_id: ID of the document
            chunks: List of text chunks
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in batch
        embeddings = self.generate_embeddings_batch(chunks)
        
        # Prepare data for batch insert
        chunks_data = [
            (chunk, idx, embedding) 
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Store in database
        print("Storing chunks in database...")
        db.insert_chunks_batch(document_id, chunks_data)
        
        print(f"Successfully stored {len(chunks)} chunks")
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 3, 
        document_id: int = None
    ) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            document_id: Optional - limit search to specific document
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search in vector database
        results = db.similarity_search(query_embedding, top_k, document_id)
        
        return results
    
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Dict],
        min_similarity: float = 0.5
    ) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User question
            context_chunks: Retrieved relevant chunks
            min_similarity: Minimum similarity threshold
            
        Returns:
            Generated answer
        """
        # Check if we have relevant context
        if not context_chunks:
            return "I couldn't find any relevant information in the document to answer your question."
        
        # Check similarity scores
        max_similarity = max([chunk['similarity'] for chunk in context_chunks])
        
        if max_similarity < min_similarity:
            return f"I found some information but it doesn't seem very relevant to your question (confidence: {max_similarity:.2f}). Please try rephrasing your question."
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Chunk {i+1}]: {chunk['chunk_text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create prompt for LLM
        prompt = f"""You are a helpful assistant answering questions based on the provided document context.

Context from the document:
{context}

User Question: {query}

Instructions:
- Answer the question using ONLY the information from the provided context
- If the context doesn't contain enough information to fully answer the question, say so
- Be concise and specific
- If you're making inferences, clearly indicate that
- Do not make up information that's not in the context

Answer:"""
        
        try:
            # Generate response using Gemini (new API)
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',  # Use flash for speed (can also use gemini-1.5-pro)
                contents=prompt
            )
            answer = response.text
            
            return answer
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(
        self, 
        question: str, 
        document_id: int = None, 
        top_k: int = 3
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate
        
        Args:
            question: User question
            document_id: Optional - limit to specific document
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        print(f"Retrieving relevant chunks for: '{question}'")
        chunks = self.retrieve_relevant_chunks(question, top_k, document_id)
        
        # Generate answer
        print("Generating answer...")
        answer = self.generate_answer(question, chunks)
        
        # Prepare response
        response = {
            "answer": answer,
            "sources": [
                {
                    "text": chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'],
                    "similarity": float(chunk['similarity']),
                    "filename": chunk['filename']
                }
                for chunk in chunks
            ]
        }
        
        return response


# Singleton instance
rag_service = RAGService()