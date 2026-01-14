import os
import json
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import OPENAI_EMBED_MODEL, OPENAI_API_KEY, VECTORSTORE_DIR, PROCESSED_DIR
from rag.ingest import ingest_all

def build_index() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    chunks = ingest_all()

    # save a small manifest for debugging/observability
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_chunks": len(chunks),
                "example_metadata": chunks[0].metadata if chunks else {},
            },
            f,
            indent=2,
        )

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORSTORE_DIR)

    print(f" Built FAISS index with {len(chunks)} chunks at: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    build_index()