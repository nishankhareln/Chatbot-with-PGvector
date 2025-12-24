Here , we need to create a folders structure at the beginning 
1.backend - which consists of listed files
2.frontend - which contanins basic streamlit face .

In backend folder, we had different files:

a.database.py - This file connects Connects to PostgreSQL
                    -Stores uploaded documents
                    -Breaks documents into chunks
                    -Stores vector embeddings
                    -Performs semantic similarity search

        RealDictCursor-Returns query results as dictionaries instead of tuples
        DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:2060@localhost:5432/rag_database"
)
This part you need to add the username and password.  postgresql://username:password@host:port/database_name
Here, we had a methods:
I.init -intialize database
II.parse_db_url()-convert string into dictionary
III.get connection()-creates a postgresql connection
IV.insert_document - insert document metadata and binary file data
V.insert_chunk - stores chunk and convert it into vector embeddings
VI.similarity search  - find most relevant chunks using cosine similarity
VII.get_latest_document_id()-Fetches the most recently uploaded document ID
IX.db = Database()


create and run the Postgres Docker container. Then I mount the init.sql file, which is automatically executed by the container on first startup. This ensures the database and required tables are initialized correctly.”

--------------------------------------------------------------------------------------------------------------
The backend is a RAG (Retrieval Augmented Generation) system that:

1.Accepts document uploads (PDF, TXT, Markdown)
2.Processes and stores documents in a vector database
3.Answers questions based on document content
4.Returns answers with source citations

Technology Stack:

FastAPI: Web framework for APIs
PostgreSQL + pgvector: Vector database
LangChain: Document processing
Sentence Transformers: Text embeddings
Google Gemini: LLM for answer generation


┌─────────────┐
│   main.py   │  ← API endpoints (FastAPI)
└──────┬──────┘
       │
       ├─→ ┌──────────────────┐
       │   │  database.py     │  ← PostgreSQL operations
       │   └──────────────────┘
       │
       ├─→ ┌──────────────────┐
       │   │document_service.py│ ← Load & chunk documents
       │   └──────────────────┘
       │
       └─→ ┌──────────────────┐
           │  rag_service.py  │  ← Embeddings & LLM
           └──────────────────┘

database.py-
Here, Encapsulates all database operations as methods.
Constructor: __init__()
Why Context Manager?
-Prevents connection leaks
-Ensures proper error handling
-Cleaner code


Method: insert_document() - Store Document

psycopg2.Binary()?

Safely converts Python bytes to PostgreSQL BYTEA
------------------------------
Method: similarity_search() - Vector Search

What it does:

Takes question embedding as input
Compares with ALL chunk embeddings in database
Uses pgvector's <=> operator (cosine distance)
Returns top K most similar chunks

The Magic Operator: <=>

<=> is pgvector's cosine distance operator
Distance = 0 means identical
Distance = 2 means opposite
We convert to similarity: similarity = 1 - distance


Now in documents.py:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from typing import List

Imports:
RecursiveCharacterTextSplitter: Splits text intelligently
PyPDFLoader: Reads PDF files
TextLoader: Reads text files

rag_service.py:
SentenceTransformer: Creates embeddings
genai: Google Gemini API
db: Database instance

Constructor: __init__()
pythonclass RAGService:
def __init__(self):
    """Initialize embedding model and LLM"""
    print("Loading embedding model...")
    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    self.llm = genai.GenerativeModel('gemini-pro')
    
    print("RAG Service initialize")

What it does:

Loads SentenceTransformer model (80MB download first time)
Initializes Gemini Pro model
Both cached in memory for fast repeated use

Model Details:

all-MiniLM-L6-v2: 384-dimensional embeddings, trained on 1B+ pairs
gemini-pro: Google's free LLM tier
