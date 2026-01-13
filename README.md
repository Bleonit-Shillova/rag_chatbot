# Genpact RAG Chatbot (LangChain + FAISS + OpenAI)

## Features
- Ingestion: load -> clean -> chunk -> embed -> index (FAISS)
- Retrieval: hybrid (BM25 + vectors), Top-K
- Answers grounded in documents + citations
- Safe "I don't know"
- Metadata filter by industry
- Prompt-injection defense (basic)

## Setup (Mac / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY