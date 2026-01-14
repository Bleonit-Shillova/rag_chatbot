import os
import sys
from pathlib import Path
import mimetypes
import json
import uuid  

# Ensure imports work when Streamlit runs from /app
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from rag.qa import answer_question
from rag.retrieval import load_vectorstore  # keep this at top-level

# ---------- Paths for source linking ----------
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)
RAW_DIR = PROJECT_ROOT_PATH / "data" / "raw"
CHAT_SAVE_PATH = PROJECT_ROOT_PATH / "data" / "processed" / "chat_history.json"


def resolve_source_path(source: str) -> Path | None:
    """
    source looks like: 'banking/bai_banking_outlook_2024_executive_report.pdf'
    """
    if not source:
        return None
    p = (RAW_DIR / source).resolve()
    # Safety: ensure the resolved path is still under data/raw
    try:
        p.relative_to(RAW_DIR.resolve())
    except Exception:
        return None
    return p if p.exists() else None


def file_bytes_and_type(path: Path):
    data = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    return data, (mime or "application/octet-stream")


def load_chat_from_disk():
    if CHAT_SAVE_PATH.exists():
        try:
            st.session_state.chat = json.loads(CHAT_SAVE_PATH.read_text(encoding="utf-8"))
        except Exception:
            st.session_state.chat = []


def save_chat_to_disk():
    CHAT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAT_SAVE_PATH.write_text(json.dumps(st.session_state.chat, indent=2), encoding="utf-8")



def render_citations_grouped(citations, show_chunk_preview, all_docs_for_bm25, scope_key: str):
    if not citations:
        return

    st.subheader("Sources & Citations")

    for i, c in enumerate(citations, start=1):
        src = c.get("source")
        ind = c.get("industry")
        refs = c.get("references", [])

        with st.container(border=True):
            st.markdown(f"**[{i}] {src}**")
            st.caption(f"Industry: {ind}")

            # Show all referenced page/chunks for this source
            if refs:
                st.markdown("**Referenced sections:**")
                for r in refs:
                    st.markdown(f"- Page {r.get('page')} (chunk {r.get('chunk_id')})")

            # Download/Open button for local doc
            p = resolve_source_path(src)
            if p:
                data, mime = file_bytes_and_type(p)
                st.download_button(
                    label="⬇️ Download / Open source file",
                    data=data,
                    file_name=p.name,
                    mime=mime,
                    
                    key=f"dl_{scope_key}_{hash(src)}_{i}",
                    use_container_width=True,
                )
            else:
                st.warning("Source file not found locally under data/raw. (Not committed to Git is normal.)")

            # Optional: show chunk preview text if you enable it
            if show_chunk_preview and refs:
                # Show preview for the FIRST reference to keep UI clean
                first_ref = refs[0]
                chunk_id = first_ref.get("chunk_id")

                matches = [
                    d for d in all_docs_for_bm25
                    if d.metadata.get("source") == src and d.metadata.get("chunk_id") == chunk_id
                ]
                if matches:
                    preview = matches[0].page_content
                    with st.expander("View retrieved chunk text (first reference)"):
                        st.write(preview[:2500] + ("..." if len(preview) > 2500 else ""))
                else:
                    st.caption("Chunk preview not available for this citation.")


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Genpact RAG Chatbot", layout="wide")
st.title("Genpact RAG Chatbot (LangChain + FAISS + OpenAI)")

with st.sidebar:
    st.header("Filters")
    industry = st.text_input("Industry filter (optional)", value="").strip().lower() or None
    st.caption(
        "Example: banking, insurance, healthcare, lifesciences, manufacturing, comms, energy, retail, privateequity"
    )
    st.divider()
    st.header("Index status")
    st.write("Make sure you've run: `python -m rag.index`")
    st.divider()
    show_chunk_preview = st.checkbox("Show retrieved chunk preview", value=False)
    persist_chat = st.checkbox("Persist chat across refresh", value=True)

# Chat is now a list of dicts: {"id":..., "q":..., "answer":..., "citations":[...]}
if "chat" not in st.session_state:
    st.session_state.chat = []
    if persist_chat:
        load_chat_from_disk()


for t in st.session_state.chat:
    if "id" not in t:
        t["id"] = str(uuid.uuid4())

@st.cache_resource
def load_docs_for_bm25():
   
    vs = load_vectorstore()
    return list(vs.docstore._dict.values())

all_docs_for_bm25 = load_docs_for_bm25()

# Render chat history (including citations)
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_citations_grouped(
            turn.get("citations", []),
            show_chunk_preview,
            all_docs_for_bm25,
            scope_key=f"hist_{turn['id']}",
        )

q = st.chat_input("Ask a question about the documents...")

if q:
    with st.chat_message("user"):
        st.write(q)

    # chat_history should be List[Tuple[q, answer]]
    chat_pairs = [(t["q"], t["answer"]) for t in st.session_state.chat]

    result = answer_question(
        question=q,
        all_docs_for_bm25=all_docs_for_bm25,
        industry_filter=industry,
        chat_history=chat_pairs,
    )

    with st.chat_message("assistant"):
        st.write(result["answer"])
        render_citations_grouped(
            result.get("citations", []),
            show_chunk_preview,
            all_docs_for_bm25,
            scope_key="current",
        )

    st.session_state.chat.append(
        {
            "id": str(uuid.uuid4()),  
            "q": q,
            "answer": result["answer"],
            "citations": result.get("citations", []),
        }
    )

    if persist_chat:
        save_chat_to_disk()