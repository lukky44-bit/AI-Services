import os
from dotenv import load_dotenv

# Load environment variables from absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, ".env")
load_dotenv(env_path)

# Resolve absolute path to the rag_agent directory
current_dir = os.path.dirname(os.path.abspath(__file__))

class Config:
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # K6 Expert Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(current_dir, "k6_chroma_db"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
