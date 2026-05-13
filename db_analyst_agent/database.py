import psycopg2
from .config import Config

class DatabaseManager:
    """Manages connection to the PostgreSQL database."""
    
    def get_connection(self):
        """Creates and returns a new psycopg2 connection."""
        return psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
