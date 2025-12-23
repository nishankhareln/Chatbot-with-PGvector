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



