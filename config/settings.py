import os 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

CHROMA_DB_PATH: str = str(
    Path(__file__).resolve().parent.parent / "chroma_db"
)

CHROMA_COLLECTION_NAME: str = "research_papers"

CHUNK_SIZE_WORDS: int = 500
CHUNK_OVERLAP_WORDS: int = 50
UPLOADED_PDFS_DIR: str = str(
    Path(__file__).resolve().parent.parent / "uploaded_pdfs"
)

TOP_K_RESULTS: int = 4

MAX_SUB_QUESTIONS: int = 4