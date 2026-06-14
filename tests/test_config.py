"""
test_config.py — Unit tests for backend/config.py.

Verifies all configuration constants have the correct types and sensible
values, and that the API_KEY falls back to the default when the env var is
absent.
"""

import os
import importlib

import pytest
import backend.config as cfg


class TestDirectoryPaths:
    def test_upload_dir_is_string(self):
        assert isinstance(cfg.UPLOAD_DIR, str)

    def test_upload_dir_non_empty(self):
        assert len(cfg.UPLOAD_DIR) > 0

    def test_vector_db_dir_is_string(self):
        assert isinstance(cfg.VECTOR_DB_DIR, str)

    def test_vector_db_dir_non_empty(self):
        assert len(cfg.VECTOR_DB_DIR) > 0


class TestEmbeddingModel:
    def test_embedding_model_is_string(self):
        assert isinstance(cfg.EMBEDDING_MODEL, str)

    def test_embedding_model_non_empty(self):
        assert len(cfg.EMBEDDING_MODEL) > 0

    def test_embedding_model_contains_model_name(self):
        # Should reference a HuggingFace model path
        assert "/" in cfg.EMBEDDING_MODEL


class TestLLMConfig:
    def test_llm_model_name_is_string(self):
        assert isinstance(cfg.LLM_MODEL_NAME, str)

    def test_available_models_is_list(self):
        assert isinstance(cfg.AVAILABLE_MODELS, list)

    def test_available_models_non_empty(self):
        assert len(cfg.AVAILABLE_MODELS) > 0

    def test_available_models_all_strings(self):
        assert all(isinstance(m, str) for m in cfg.AVAILABLE_MODELS)

    def test_default_model_in_available_models(self):
        assert cfg.DEFAULT_MODEL in cfg.AVAILABLE_MODELS


class TestChunkingParams:
    def test_chunk_size_is_positive_int(self):
        assert isinstance(cfg.CHUNK_SIZE, int)
        assert cfg.CHUNK_SIZE > 0

    def test_chunk_overlap_is_non_negative_int(self):
        assert isinstance(cfg.CHUNK_OVERLAP, int)
        assert cfg.CHUNK_OVERLAP >= 0

    def test_chunk_overlap_less_than_chunk_size(self):
        assert cfg.CHUNK_OVERLAP < cfg.CHUNK_SIZE


class TestRetrievalAndGeneration:
    def test_top_k_is_positive_int(self):
        assert isinstance(cfg.TOP_K_RESULTS, int)
        assert cfg.TOP_K_RESULTS > 0

    def test_max_new_tokens_is_positive_int(self):
        assert isinstance(cfg.MAX_NEW_TOKENS, int)
        assert cfg.MAX_NEW_TOKENS > 0

    def test_temperature_is_float_in_range(self):
        assert isinstance(cfg.TEMPERATURE, float)
        assert 0.0 <= cfg.TEMPERATURE <= 2.0


class TestAllowedExtensions:
    def test_allowed_extensions_is_list(self):
        assert isinstance(cfg.ALLOWED_EXTENSIONS, list)

    def test_allowed_extensions_non_empty(self):
        assert len(cfg.ALLOWED_EXTENSIONS) > 0

    def test_all_extensions_start_with_dot(self):
        for ext in cfg.ALLOWED_EXTENSIONS:
            assert ext.startswith("."), f"Extension '{ext}' should start with '.'"

    def test_common_types_present(self):
        for expected in [".pdf", ".txt"]:
            assert expected in cfg.ALLOWED_EXTENSIONS


class TestAuthentication:
    def test_api_key_is_non_empty_string(self):
        assert isinstance(cfg.API_KEY, str)
        assert len(cfg.API_KEY) > 0

    def test_api_key_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("RAG_API_KEY", "my-test-key-xyz")
        # Re-import config to pick up new env var
        import importlib
        reloaded = importlib.reload(cfg)
        assert reloaded.API_KEY == "my-test-key-xyz"

    def test_api_key_has_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("RAG_API_KEY", raising=False)
        import importlib
        reloaded = importlib.reload(cfg)
        assert len(reloaded.API_KEY) > 0  # default is non-empty


class TestApiAndPreview:
    def test_api_base_url_is_string(self):
        assert isinstance(cfg.API_BASE_URL, str)

    def test_api_base_url_starts_with_http(self):
        assert cfg.API_BASE_URL.startswith("http")

    def test_preview_char_limit_is_positive_int(self):
        assert isinstance(cfg.PREVIEW_CHAR_LIMIT, int)
        assert cfg.PREVIEW_CHAR_LIMIT > 0
