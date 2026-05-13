import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables just in case
load_dotenv()

class DatabaseManager:
    """Manages connection to the PostgreSQL database."""
    
    def __init__(self):
        # Load connection details from environment variables with defaults
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.database = os.getenv("DB_NAME", "ptbot")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "password")

    def get_connection(self):
        """Creates and returns a new psycopg2 connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
