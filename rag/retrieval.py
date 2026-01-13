from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.config import VECTORSTORE_DIR, OPENAI_EMBED_MODEL, TOP_K, MIN_RELEVANCE


def load_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)


def filter_docs(docs: List[Document], industry_filter: Optional[str]) -> List[Document]:
    if not industry_filter:
        return docs
    return [d for d in docs if (d.metadata.get("industry") == industry_filter)]


def format_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    cites = []
    for d in docs:
        cites.append(
            {
                "source": d.metadata.get("source"),
                "industry": d.metadata.get("industry"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
        )
    return cites


def is_prompt_injection(text: str) -> bool:
    red_flags = [
        "ignore previous instructions",
        "system prompt",
        "developer message",
        "exfiltrate",
        "api key",
        "password",
    ]
    t = text.lower()
    return any(r in t for r in red_flags)


def _doc_key(d: Document) -> str:
    # stable identity for merging results
    return f"{d.metadata.get('source')}|{d.metadata.get('page')}|{d.metadata.get('chunk_id')}|{d.metadata.get('start_index')}"


def rrf_merge(bm25_docs: List[Document], vec_docs: List[Document], k: int = TOP_K, rrf_k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion:
    score(doc) = sum_i 1 / (rrf_k + rank_i)
    """
    scores = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for rank, d in enumerate(bm25_docs, start=1):
        key = _doc_key(d)
        scores[key] += 1.0 / (rrf_k + rank)
        doc_map[key] = d

    for rank, d in enumerate(vec_docs, start=1):
        key = _doc_key(d)
        scores[key] += 1.0 / (rrf_k + rank)
        doc_map[key] = d

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = [doc_map[key] for key, _ in ranked[:k]]
    return out


def retrieve(
    query: str,
    all_docs_for_bm25: List[Document],
    industry_filter: Optional[str] = None,
) -> Tuple[List[Document], List[Dict[str, Any]]]:

    if is_prompt_injection(query):
        return [], []

    # BM25 retrieval (lexical)
    bm25_corpus = filter_docs(all_docs_for_bm25, industry_filter)
    bm25 = BM25Retriever.from_documents(bm25_corpus)
    bm25.k = TOP_K
    bm25_docs = bm25.get_relevant_documents(query)

    # Vector retrieval (semantic)
    vs = load_vectorstore()

    # We use similarity_search for docs and similarity_search_with_relevance_scores for thresholding
    vec_docs = vs.similarity_search(query, k=TOP_K * 2)
    vec_docs = filter_docs(vec_docs, industry_filter)

    scored = vs.similarity_search_with_relevance_scores(query, k=TOP_K * 2)
    best = 0.0
    for d, score in scored:
        if (not industry_filter) or d.metadata.get("industry") == industry_filter:
            best = max(best, float(score))

    if best < MIN_RELEVANCE:
        return [], []

    # Hybrid merge
    merged = rrf_merge(bm25_docs, vec_docs, k=TOP_K)
    return merged, format_citations(merged)