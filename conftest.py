"""
conftest.py — Shared pytest fixtures and mock patches for the RAG Q&A System.

Heavy ML dependencies (sentence-transformers, faiss-cpu, torch) are stubbed
out at the module level so the test suite can run in CI without downloading
gigabytes of model weights.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Stub out heavy ML libraries before any backend module is imported
# ─────────────────────────────────────────────────────────────────────────────

def _make_module(name: str) -> types.ModuleType:
    """Return a new, empty stub module registered in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# torch — not installed in CI
if "torch" not in sys.modules:
    _make_module("torch")

# sentence_transformers
if "sentence_transformers" not in sys.modules:
    st = _make_module("sentence_transformers")
    st.SentenceTransformer = MagicMock()

# faiss — not installed in CI
if "faiss" not in sys.modules:
    _make_module("faiss")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary plain-text file for document-loader tests."""
    p = tmp_path / "sample.txt"
    p.write_text(
        "The quick brown fox jumps over the lazy dog. " * 40,
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def sample_md_file(tmp_path):
    """Create a temporary Markdown file for document-loader tests."""
    p = tmp_path / "sample.md"
    p.write_text(
        "# Heading\n\nThis is a **markdown** document.\n\n" + ("paragraph " * 60),
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file for document-loader tests."""
    p = tmp_path / "data.csv"
    p.write_text(
        "id,name,value\n1,Alice,100\n2,Bob,200\n3,Carol,300\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Return a FastAPI TestClient with storage dirs redirected to tmp_path.

    All heavy ML backends are mocked so no model weights are downloaded.
    """
    import importlib
    import os

    upload_dir = str(tmp_path / "uploads")
    vector_dir = str(tmp_path / "vector_db")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)

    monkeypatch.setenv("RAG_API_KEY", "test-key")

    # Patch config constants before the app is used
    import backend.config as cfg
    monkeypatch.setattr(cfg, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(cfg, "VECTOR_DB_DIR", vector_dir)
    monkeypatch.setattr(cfg, "API_KEY", "test-key")

    from unittest.mock import MagicMock, patch

    embeddings_mock = MagicMock()
    chunk_mock = MagicMock()
    chunk_mock.page_content = "chunk"
    chunk_mock.metadata = {"source": "f.txt"}

    with (
        patch("backend.vector_store.get_embeddings", return_value=embeddings_mock),
        patch("backend.main.load_vector_store", return_value=None),
        patch("backend.main.add_to_vector_store"),
        patch("backend.main.load_and_split", return_value=[chunk_mock]),
    ):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            client = TestClient(app, raise_server_exceptions=True)
        except RuntimeError as exc:
            if "python-multipart" in str(exc) or "multipart" in str(exc).lower():
                pytest.skip(
                    "python-multipart is not properly installed. "
                    "Run: pip uninstall multipart && pip install python-multipart"
                )
            raise

        yield client, upload_dir, vector_dir

