"""
vector_store.py — FAISS index manager.

Provides helpers to create, load, and incrementally update a FAISS-backed
vector store persisted to disk.
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from backend.config import VECTOR_DB_DIR
from backend.embeddings import get_embeddings


def create_vector_store(docs: List[Document]) -> FAISS:
    """Create a new FAISS index from document chunks and persist it.

    Args:
        docs: List of ``Document`` chunks to embed and store.

    Returns:
        The newly created ``FAISS`` vector store object.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_DB_DIR)

    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """Load an existing FAISS index from disk.

    Returns:
        The loaded ``FAISS`` vector store, or ``None`` if the index
        directory does not exist or is empty.
    """
    index_file = os.path.join(VECTOR_DB_DIR, "index.faiss")

    if not os.path.exists(index_file):
        return None

    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def add_to_vector_store(new_docs: List[Document]) -> FAISS:
    """Add new document chunks to the vector store.

    If an existing index is found on disk it is loaded first and the new
    documents are merged in.  Otherwise a fresh index is created.  The
    updated index is persisted before returning.

    Args:
        new_docs: List of new ``Document`` chunks to add.

    Returns:
        The updated ``FAISS`` vector store object.
    """
    existing_store = load_vector_store()

    if existing_store is not None:
        embeddings = get_embeddings()
        new_store = FAISS.from_documents(new_docs, embeddings)
        existing_store.merge_from(new_store)

        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        existing_store.save_local(VECTOR_DB_DIR)
        return existing_store

    return create_vector_store(new_docs)
