"""
core/indexing.py
────────────────
Takes the chunks produced by ingestion.py and stores them in ChromaDB.
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
)

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"  Loading embedding model '{EMBEDDING_MODEL}' ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("  Embedding model ready.")
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_and_index(chunks: List[Dict[str, Any]]) -> int:
    """
    Embed a list of chunk dicts and upsert them into ChromaDB.
    Uses upsert so re-uploading the same PDF is idempotent.
    """
    if not chunks:
        print("  No chunks to index.")
        return 0

    model      = _get_embedding_model()
    collection = _get_collection()

    texts       = [c["text"]        for c in chunks]
    paper_names = [c["paper_name"]  for c in chunks]
    page_nums   = [c["page_number"] for c in chunks]

    ids = [
        f"{paper_names[i]}_p{page_nums[i]}_c{i}"
        for i in range(len(chunks))
    ]

    print(f"  Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    metadatas = [
        {"paper_name": paper_names[i], "page_number": page_nums[i]}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"  Indexed {len(chunks)} chunks into '{CHROMA_COLLECTION_NAME}'.")
    return len(chunks)


def get_indexed_papers() -> List[str]:
    try:
        collection = _get_collection()
        result = collection.get(include=["metadatas"])
        names = {m["paper_name"] for m in result["metadatas"] if m}
        return sorted(names)
    except Exception:
        return []


def get_collection_stats() -> Dict[str, Any]:
    try:
        collection = _get_collection()
        count  = collection.count()
        papers = get_indexed_papers()
        return {
            "total_chunks": count,
            "total_papers": len(papers),
            "papers":       papers,
        }
    except Exception as e:
        return {"error": str(e)}