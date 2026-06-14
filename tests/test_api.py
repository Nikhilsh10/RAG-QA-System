"""
test_api.py — Integration tests for the FastAPI application (backend/main.py).

All heavy backends (FAISS, embeddings, Ollama) are mocked so the tests run
without any ML dependencies.  The FastAPI TestClient is used to exercise the
full HTTP request/response cycle.

Endpoints covered:
- GET  /health
- GET  /models
- GET  /documents
- POST /upload     (valid file, unsupported extension, missing vector store handling)
- POST /ask        (no vector store, with vector store via mock)
- DELETE /clear
"""

import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (supplement those in conftest.py)
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = "test-key"
AUTH_HEADERS = {"X-API-Key": API_KEY}


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, api_client):
        client, *_ = api_client
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self, api_client):
        client, *_ = api_client
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_requires_no_auth(self, api_client):
        """Health check must work even without API key."""
        client, *_ = api_client
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_unauthorized_other_endpoint_without_key(self, api_client):
        client, *_ = api_client
        resp = client.get("/models")  # no key
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /models
# ─────────────────────────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_returns_200(self, api_client):
        client, *_ = api_client
        resp = client.get("/models", headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_returns_models_list(self, api_client):
        client, *_ = api_client
        resp = client.get("/models", headers=AUTH_HEADERS)
        body = resp.json()
        assert "models" in body
        assert isinstance(body["models"], list)

    def test_models_list_non_empty(self, api_client):
        client, *_ = api_client
        resp = client.get("/models", headers=AUTH_HEADERS)
        assert len(resp.json()["models"]) > 0

    def test_models_are_strings(self, api_client):
        client, *_ = api_client
        resp = client.get("/models", headers=AUTH_HEADERS)
        for m in resp.json()["models"]:
            assert isinstance(m, str)


# ─────────────────────────────────────────────────────────────────────────────
# /documents (list)
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentsEndpoint:
    def test_returns_200_when_dir_empty(self, api_client):
        client, upload_dir, _ = api_client
        resp = client.get("/documents", headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_empty_upload_dir_gives_empty_list(self, api_client):
        client, upload_dir, _ = api_client
        # Ensure dir is clean
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))
        resp = client.get("/documents", headers=AUTH_HEADERS)
        assert resp.json()["documents"] == []

    def test_uploaded_files_appear_in_list(self, api_client):
        from unittest.mock import patch
        client, upload_dir, _ = api_client
        # Create a file in the upload dir and patch the endpoint's UPLOAD_DIR to match
        import os
        os.makedirs(upload_dir, exist_ok=True)
        open(os.path.join(upload_dir, "test.txt"), "w").close()
        with patch("backend.main.UPLOAD_DIR", upload_dir):
            resp = client.get("/documents", headers=AUTH_HEADERS)
        assert "test.txt" in resp.json()["documents"]

    def test_documents_list_is_sorted(self, api_client):
        from unittest.mock import patch
        client, upload_dir, _ = api_client
        import os
        for name in ["zebra.txt", "alpha.txt", "mango.txt"]:
            open(os.path.join(upload_dir, name), "w").close()
        with patch("backend.main.UPLOAD_DIR", upload_dir):
            resp = client.get("/documents", headers=AUTH_HEADERS)
        docs = resp.json()["documents"]
        assert docs == sorted(docs)


# ─────────────────────────────────────────────────────────────────────────────
# /upload
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def test_upload_valid_txt_returns_200(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("hello.txt", BytesIO(b"hello world " * 20), "text/plain")},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200

    def test_upload_response_has_expected_fields(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("doc.txt", BytesIO(b"content " * 30), "text/plain")},
            headers=AUTH_HEADERS,
        )
        body = resp.json()
        assert "message" in body
        assert "filename" in body
        assert "chunks" in body

    def test_upload_filename_echoed_back(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("myfile.txt", BytesIO(b"data " * 20), "text/plain")},
            headers=AUTH_HEADERS,
        )
        assert resp.json()["filename"] == "myfile.txt"

    def test_upload_chunks_is_positive_int(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("chunk_test.txt", BytesIO(b"text " * 40), "text/plain")},
            headers=AUTH_HEADERS,
        )
        assert resp.json()["chunks"] >= 1

    def test_upload_unsupported_extension_returns_400(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("bad.exe", BytesIO(b"binary"), "application/octet-stream")},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_upload_unsupported_extension_error_message(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("bad.exe", BytesIO(b"binary"), "application/octet-stream")},
            headers=AUTH_HEADERS,
        )
        assert "Unsupported" in resp.json()["detail"] or "unsupported" in resp.json()["detail"]

    def test_upload_without_auth_returns_401(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("note.txt", BytesIO(b"text"), "text/plain")},
        )
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /ask
# ─────────────────────────────────────────────────────────────────────────────

class TestAskEndpoint:
    def test_ask_without_vector_store_returns_400(self, api_client):
        """load_vector_store returns None → no documents indexed."""
        client, *_ = api_client
        # api_client fixture patches load_vector_store to return None
        resp = client.post(
            "/ask",
            json={"question": "What is RAG?"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_ask_error_message_mentions_documents(self, api_client):
        client, *_ = api_client
        resp = client.post(
            "/ask",
            json={"question": "What is RAG?"},
            headers=AUTH_HEADERS,
        )
        detail = resp.json().get("detail", "").lower()
        assert "document" in detail or "index" in detail

    def test_ask_with_vector_store_returns_200(self, api_client):
        """Patch load_vector_store to return a real-looking mock."""
        from unittest.mock import MagicMock, patch
        from langchain.schema import Document

        client, *_ = api_client
        mock_vs = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "RAG is retrieval-augmented generation.",
            "source_documents": [],
        }

        with (
            patch("backend.main.load_vector_store", return_value=mock_vs),
            patch("backend.main.build_qa_chain", return_value=mock_chain),
        ):
            resp = client.post(
                "/ask",
                json={"question": "What is RAG?"},
                headers=AUTH_HEADERS,
            )

        assert resp.status_code == 200

    def test_ask_response_has_answer_sources_chunks(self, api_client):
        from unittest.mock import MagicMock, patch

        client, *_ = api_client
        mock_vs = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "An answer.",
            "source_documents": [],
        }

        with (
            patch("backend.main.load_vector_store", return_value=mock_vs),
            patch("backend.main.build_qa_chain", return_value=mock_chain),
        ):
            resp = client.post(
                "/ask",
                json={"question": "?"},
                headers=AUTH_HEADERS,
            )

        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "chunks" in body

    def test_ask_without_auth_returns_401(self, api_client):
        client, *_ = api_client
        resp = client.post("/ask", json={"question": "test?"})
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /clear
# ─────────────────────────────────────────────────────────────────────────────

class TestClearEndpoint:
    def test_clear_returns_200(self, api_client):
        client, *_ = api_client
        resp = client.delete("/clear", headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_clear_response_has_message(self, api_client):
        client, *_ = api_client
        resp = client.delete("/clear", headers=AUTH_HEADERS)
        assert "message" in resp.json()

    def test_clear_empties_upload_dir(self, api_client):
        from unittest.mock import patch
        import shutil
        client, upload_dir, vector_dir = api_client
        # Place a sentinel file in our dir
        open(os.path.join(upload_dir, "sentinel.txt"), "w").close()
        # Patch both UPLOAD_DIR and VECTOR_DB_DIR so the endpoint clears the right dirs
        with (
            patch("backend.main.UPLOAD_DIR", upload_dir),
            patch("backend.main.VECTOR_DB_DIR", vector_dir),
        ):
            client.delete("/clear", headers=AUTH_HEADERS)
        remaining = [
            f for f in os.listdir(upload_dir)
            if os.path.isfile(os.path.join(upload_dir, f))
        ]
        assert remaining == []

    def test_clear_without_auth_returns_401(self, api_client):
        client, *_ = api_client
        resp = client.delete("/clear")
        assert resp.status_code == 401
