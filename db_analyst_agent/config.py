import os
from dotenv import load_dotenv

# Load environment variables from absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, ".env")
load_dotenv(env_path)

class Config:
    """Central configuration for the Database Analyst Agent."""
    
    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "ptbot")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    
    # LLM Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
