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


def infer_industries_from_query(query: str) -> List[str]:
    """
    Heuristic mapper from query text -> your metadata industry labels.
    Only used when user did NOT set an explicit industry_filter.
    """
    q = query.lower()
    inds: List[str] = []

    # Banking
    if any(w in q for w in ["bank", "banking", "capital markets", "credit union"]):
        inds.append("banking")

    # Insurance
    if any(w in q for w in ["insurer", "insurers", "insurance", "underwriting", "claims"]):
        inds.append("insurance")

    # Healthcare
    if any(w in q for w in ["healthcare", "hospital", "payer", "provider", "medicare", "medicaid"]):
        inds.append("healthcare")

    # Life sciences
    if any(w in q for w in ["life sciences", "lifesciences", "pharma", "biotech", "medtech"]):
        inds.append("lifesciences")

    # Manufacturing
    if any(w in q for w in ["manufacturing", "factory", "supply chain", "plant"]):
        inds.append("manufacturing")

    # High tech
    if any(w in q for w in ["high tech", "hightech", "semiconductor", "chip", "electronics"]):
        inds.append("hightech")

    # Comms / telecom
    if any(w in q for w in ["telecom", "telco", "communications", "comms"]):
        inds.append("comms")

    # Energy
    if any(w in q for w in ["energy", "oil", "gas", "utilities", "power"]):
        inds.append("energy")

    # Retail
    if any(w in q for w in ["retail", "store", "e-commerce", "ecommerce"]):
        inds.append("retail")

    # Private equity
    if any(w in q for w in ["private equity", "privateequity", "pe firm"]):
        inds.append("privateequity")

    # Consumer categories (only if explicitly mentioned)
    if any(w in q for w in ["consumer tech", "consumertech"]):
        inds.append("consumertech")

    if any(w in q for w in ["consumer goods", "consumergoods", "cpg"]):
        inds.append("consumergoods")

    # Software (only if explicitly mentioned)
    if any(w in q for w in ["software", "saas"]):
        inds.append("software")

    # de-dupe preserving order
    return list(dict.fromkeys(inds))


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
    return (
        f"{d.metadata.get('source')}|"
        f"{d.metadata.get('page')}|"
        f"{d.metadata.get('chunk_id')}|"
        f"{d.metadata.get('start_index')}"
    )


def rrf_merge(
    bm25_docs: List[Document],
    vec_docs: List[Document],
    k: int = TOP_K,
    rrf_k: int = 60,
) -> List[Document]:
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

    # If user didn't set explicit filter, but query mentions industries, restrict to them
    inferred_industries = infer_industries_from_query(query) if not industry_filter else []

    def allow_doc(d: Document) -> bool:
        if industry_filter:
            return d.metadata.get("industry") == industry_filter
        if inferred_industries:
            return d.metadata.get("industry") in inferred_industries
        return True

    # BM25 retrieval (lexical)
    if industry_filter:
        bm25_corpus = filter_docs(all_docs_for_bm25, industry_filter)
    elif inferred_industries:
        bm25_corpus = [d for d in all_docs_for_bm25 if allow_doc(d)]
    else:
        bm25_corpus = all_docs_for_bm25

    
    if not bm25_corpus:
        return [], []

    bm25 = BM25Retriever.from_documents(bm25_corpus)
    bm25.k = TOP_K
    bm25_docs = bm25.invoke(query)

    # Vector retrieval (semantic)
    vs = load_vectorstore()

    vec_docs = vs.similarity_search(query, k=TOP_K * 2)
    vec_docs = [d for d in vec_docs if allow_doc(d)]

    # Relevance threshold (use only allowed industries)
    scored = vs.similarity_search_with_relevance_scores(query, k=TOP_K * 2)
    best = 0.0
    for d, score in scored:
        if allow_doc(d):
            best = max(best, float(score))

    if best < MIN_RELEVANCE:
        return [], []

    # Hybrid merge
    merged = rrf_merge(bm25_docs, vec_docs, k=TOP_K)
    # Ensure merged also respects allow_doc (bm25 may include some edge cases)
    merged = [d for d in merged if allow_doc(d)]

    return merged, format_citations(merged)