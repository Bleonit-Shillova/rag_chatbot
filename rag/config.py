import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
VECTORSTORE_DIR = "vectorstore/faiss"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

TOP_K = 5
MIN_RELEVANCE = 0.22  # tune after you ingest (keeps "I don't know" honest)