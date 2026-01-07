import os
from psycopg2 import connect, OperationalError
from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5433))  # default 5432 if not set
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def check_connection():
    try:
        # Connect to PostgreSQL
        connection = connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print(" Successfull Connected to the PGadmin postgres")

        
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print("PostgreSQL version:", db_version[0])

        cursor.close()
        connection.close()

    except OperationalError as e:
        print(" Connection failed:")
        print(e)

if __name__ == "__main__":
    check_connection()
