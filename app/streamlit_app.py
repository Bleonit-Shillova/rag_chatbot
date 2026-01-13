import os
import sys

# Ensure imports work when Streamlit runs from /app
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from rag.ingest import ingest_all
from rag.qa import answer_question

st.set_page_config(page_title="Genpact RAG Chatbot", layout="wide")

st.title("Genpact RAG Chatbot (LangChain + FAISS + OpenAI)")

with st.sidebar:
    st.header("Filters")
    industry = st.text_input("Industry filter (optional)", value="").strip().lower() or None
    st.caption(
        "Example: banking, insurance, healthcare, lifesciences, manufacturing, hightech, comms, energy, retail, privateequity"
    )
    st.divider()
    st.header("Index status")
    st.write("Make sure you've run: `python -m rag.index`")

if "chat" not in st.session_state:
    st.session_state.chat = []

# BM25 needs docs in memory (lightweight for this project)
@st.cache_resource
def load_docs_for_bm25():
    return ingest_all()

all_docs_for_bm25 = load_docs_for_bm25()

q = st.chat_input("Ask a question about the documents...")

for u, a in st.session_state.chat:
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.write(a)

if q:
    with st.chat_message("user"):
        st.write(q)

    result = answer_question(
        question=q,
        all_docs_for_bm25=all_docs_for_bm25,
        industry_filter=industry,
        chat_history=st.session_state.chat,
    )

    with st.chat_message("assistant"):
        st.write(result["answer"])

        if result["citations"]:
            st.subheader("Citations (metadata)")
            st.table(result["citations"])

    st.session_state.chat.append((q, result["answer"]))