import os

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True) 