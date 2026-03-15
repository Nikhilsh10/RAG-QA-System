"""
config.py — Central configuration and constants for the RAG Q&A system.

All tuneable parameters live here so that no other module contains
hard-coded magic numbers or paths.
"""

import os

# ── Directory paths ──────────────────────────────────────────────────────────
UPLOAD_DIR: str = os.path.join("data", "uploaded_docs")
VECTOR_DB_DIR: str = os.path.join("vector_db")

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM (served via Ollama) ─────────────────────────────────────────────────
LLM_MODEL_NAME: str = "mistral"
AVAILABLE_MODELS: list[str] = ["mistral", "llama3", "gemma"]
DEFAULT_MODEL: str = "mistral"

# ── Text chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = 4

# ── Generation ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS: int = 512
TEMPERATURE: float = 0.2

# ── Supported file types ────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx", ".csv", ".md"]

# ── Authentication ──────────────────────────────────────────────────────────
API_KEY: str = os.getenv("RAG_API_KEY", "rag-secret-key-2024")

# ── API ──────────────────────────────────────────────────────────────────────
API_BASE_URL: str = "http://localhost:8000"

# ── Document preview ────────────────────────────────────────────────────────
PREVIEW_CHAR_LIMIT: int = 2000
