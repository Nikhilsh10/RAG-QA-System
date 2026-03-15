"""
embeddings.py — Singleton loader for the HuggingFace embedding model.

The model is instantiated once on the first call to ``get_embeddings()``
and reused for every subsequent call.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import EMBEDDING_MODEL

# Module-level cache: the model is loaded once, then reused.
_cached_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance.

    On the first invocation the model weights are downloaded (if not
    already cached locally) and loaded onto the CPU.  All subsequent
    calls return the same object.

    Returns:
        A ready-to-use ``HuggingFaceEmbeddings`` instance.
    """
    global _cached_embeddings

    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

    return _cached_embeddings
