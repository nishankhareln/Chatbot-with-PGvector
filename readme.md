To setup these all, 
You need to make a requirements.txt file and install in the terminal before that you need to 
setup and environment  - python -m venv environment_name
Then activate the environment - environment_name\Scripts\activate
Move ahead and and install - pip install -r requirements.txt  ---As my python version is 3.13.5
Create a folder structure for more clearance:
1.Backend with files such as  a.database.py
                              b.document_service.py
                              c.main.py - entry point for fastapi
                              d.rag_service.py

2.Frontend with app.py files (Using streamlit for the frontend part)

3.Create a .env file for containing gemini_api_key and database connection part:
                    # Google Gemini API Key 
                    GOOGLE_API_KEY=

                    # PostgreSQL Database Configuration
                    DB_HOST=localhost
                    DB_PORT=
                    DB_NAME=
                    DB_USER=
                    DB_PASSWORD=

4.After creating all necessary files and checking the connection with the database by using all the variables created in the .env file and run the check_database.py file and you have an output such as Postgresql version:.........

5.Since we are using the postgres pg vector form 
     - To use postgres for vector embedddings  we need to install or pull pgvector extension.
          docker pull pgvector/pgvector:16 (Make sure you had docker installed in the system )
          https://hub.docker.com/r/pgvector/pgvector/tags

6.Now we will compose all the pull images  = docker-compose up -d - it will create the volume required to hold the data and create a container (database ) with name rag_postgres.
-Check the container is set up or not - docker ps .
-we had mounted init.sql 

----------------------------------------------------------------------------------------------------
lets think of scenario You do not have init.sql (just image + empty volume)
1.Docker sees empty database volume → first run.
2.Postgres entrypoint script runs.
3.No SQL scripts found in /docker-entrypoint-initdb.d/.
4.Postgres creates only the default database (rag_database) and user (POSTGRES_USER).
5.No tables, no extensions, no indexes are created.

If you had init.sql---
1.Docker sees empty database volume → first run.
2.Postgres entrypoint script runs automatically.
3.It executes all .sql files in /docker-entrypoint-initdb.d/.
4.Your tables, extensions, indexes are created automatically.
5.Database ready → you can start querying.

----------------------------------------------------------------------------------------------------------

6.Now let us create a postgres script for creating tables and all forms .
-To enable postgres vector we need to create an extension:
                         CREATE EXTENSION IF NOT EXISTS vector;-Activates the pgvector extension in Postgres.
This allows us to store and search vector embeddings, which are numerical representations of text or documents.

-

create a document table :
               CREATE TABLE IF NOT EXISTS documents (
               id SERIAL PRIMARY KEY,
               filename VARCHAR(255) NOT NULL,
               file_type VARCHAR(50) NOT NULL,
               file_data BYTEA NOT NULL,
               file_size INTEGER,
               uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               );
               
id - Unique number for each file (auto-increment).        
filename  Name of the file, max 255 characters.                
file_type Type of file, e.g., PDF, DOCX.                       
file_data   The actual file content in binary form.               
file_size  File size in bytes.                                  
uploaded_at  When the file was uploaded. Defaults to current time.

-Create index for vector similarity search
-Create index for document lookup

After setting up all database and all docker the system run -
-----------------------------------------------------------------------------------------------
Lets create a scenario like user upload an python notes pdf and ask the question:
content =  pages about python 
Database  = document_id = ...,..this much chunk stored.

And after queurying user starts to ask :
1.What is python Dictionary?
-This thing user will type in streamlit chat 
     Line No 180- in app.py -streamlit captures it 
Then it will catch this part :
st.session_state.messages.append --line No -192

Now, Frontend Calss Backend API and HTTP request is sent back and it receives to the backend 
(/chat endpoint) :
                FastAPI receives HTTP request
               Validates question is not empty
               Calls rag_service.query()

RAG Service - It will starts querying in pipeline 
Receive relevant chunks from the retrieve_relevant_chunks method

Then it will generate the Question Embeddings :
query_embedding = array([0.234, -0.567, 0.891, ..., -0.456])  # 384 dimensions
And then it will search in PostgresSQL Vector Database using similarity search.
And then it will prepare context from Retrieved chunks
-----
context = """
[Chunk 1]: Python Dictionary

A dictionary is a built-in data structure that stores key-value pairs. Each key must be unique and immutable (like strings or numbers), while values can be any data type. Dictionaries are defined using curly braces {} and are very efficient for lookups.

Example:
student = {"name": "John", "age": 20, "grade": "A"}

[Chunk 2]: Data Structures in Python

Python provides several built-in data structures. Dictionaries are particularly useful for mapping keys to values. Unlike lists which use numeric indices, dictionaries use keys for accessing values. This makes them ideal for representing structured data.

[Chunk 3]: Dictionary Methods

Dictionaries support various methods like .get(), .keys(), .values(), and .items(). You can add new key-value pairs by assignment: dict[new_key] = value. The .get() method safely retrieves values without raising errors if the key doesn't exist.
"""
------

Now HTTP request is sent to Google Gemini APi and geminin model will read the prompt and generate the answer based only on provided context and returns text.

Finally backend returns to the frontend.

In simple flow,
1.user types question            
2. Streamlit capture
3. HTTP request to backend           
4. Generate question embedding          
5. PostgreSQL vector search (25 chunks)
6. Prepare context    
7. Gemini API call                       
8. Return response                   
9. Display in Streamlit               

-----------------------------------------------------------------
Why sentence transformers is used?
Main problem is computers cannot understand the text and if we move from the traditional matching of keyword matching then it will be limited .
So , we converted it into numbers from text so sentence transformers is used to convert the text into vectors that capture semantic meaning and context .


 from sentence_transformers import SentenceTransformer
 model = SentenceTransformer('all-MiniLM-L6-v2')







