import os
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.config import RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_one_file(path: str) -> List[Document]:
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    else:
        # HTML or txt/markdown: treat as text (good enough for this project)
        return TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()

def add_industry_metadata(docs: List[Document], industry: str, source: str) -> List[Document]:
    out = []
    for d in docs:
        md = dict(d.metadata or {})
        md["industry"] = industry
        md["source"] = source
        # page is present for PDFs; for non-PDFs we’ll default later
        out.append(Document(page_content=d.page_content, metadata=md))
    return out

def ingest_all() -> List[Document]:
    all_docs: List[Document] = []

    for industry in os.listdir(RAW_DIR):
        industry_dir = os.path.join(RAW_DIR, industry)
        if not os.path.isdir(industry_dir):
            continue

        for fname in os.listdir(industry_dir):
            path = os.path.join(industry_dir, fname)
            if not os.path.isfile(path):
                continue

            docs = load_one_file(path)
            docs = add_industry_metadata(docs, industry=industry, source=f"{industry}/{fname}")
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_docs)

    # add chunk_id for citations
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

        # standardize page for non-PDF
        if "page" not in c.metadata:
            c.metadata["page"] = None

    return chunks