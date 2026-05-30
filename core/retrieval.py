"""
core/retrieval.py
─────────────────
Queries ChromaDB independently for each sub-question.
"""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DB_PATH):
            raise RuntimeError(
                "ChromaDB not found. Please upload and index PDFs first."
            )
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def retrieve_for_question(
    question: str,
    top_k: int = TOP_K_RESULTS,
) -> List[Dict[str, Any]]:
    model      = _get_embedding_model()
    collection = _get_collection()

    if collection.count() == 0:
        print("  [Retrieval] Collection is empty.")
        return []

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "paper_name":  meta.get("paper_name", "Unknown"),
            "page_number": meta.get("page_number", 0),
            "distance":    round(dist, 4),
        })

    return chunks


def retrieve_for_all_subquestions(
    sub_questions: List[str],
    top_k: int = TOP_K_RESULTS,
) -> List[Dict[str, Any]]:
    all_results = []

    for i, question in enumerate(sub_questions, 1):
        print(f"  [Retrieval] Sub-question {i}/{len(sub_questions)}: '{question[:60]}...'")
        try:
            chunks = retrieve_for_question(question, top_k=top_k)
            print(f"    -> Retrieved {len(chunks)} chunks")
        except Exception as e:
            print(f"    -> Retrieval failed: {e}")
            chunks = []

        all_results.append({
            "sub_question": question,
            "chunks":       chunks,
        })

    return all_results


def get_retrieval_summary(retrieval_results: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(retrieval_results, 1):
        lines.append(f"Sub-question {i}: {r['sub_question']}")
        if r["chunks"]:
            for c in r["chunks"]:
                lines.append(
                    f"  - [{c['paper_name']}, p.{c['page_number']}] "
                    f"dist={c['distance']}"
                )
        else:
            lines.append("  - No chunks retrieved.")
    return "\n".join(lines)