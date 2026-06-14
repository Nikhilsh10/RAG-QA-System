"""
test_document_loader.py — Unit tests for backend/document_loader.py.

Tests cover:
- load_and_split : TXT, MD, CSV files; unsupported extensions; empty loaders
- get_document_preview : TXT, MD; char_limit truncation; unsupported ext
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import Document

from backend.document_loader import load_and_split, get_document_preview
import backend.config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_doc(content: str, source: str = "test.txt", page: int = 0) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


# ─────────────────────────────────────────────────────────────────────────────
# load_and_split
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndSplit:
    def test_raises_on_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "report.xyz"
        bad_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_and_split(str(bad_file))

    def test_txt_file_returns_list_of_documents(self, sample_txt_file):
        chunks = load_and_split(sample_txt_file)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_txt_chunks_are_document_instances(self, sample_txt_file):
        chunks = load_and_split(sample_txt_file)
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_txt_chunk_has_page_content(self, sample_txt_file):
        chunks = load_and_split(sample_txt_file)
        for chunk in chunks:
            assert isinstance(chunk.page_content, str)
            assert len(chunk.page_content) > 0

    def test_txt_chunk_metadata_has_source(self, sample_txt_file):
        chunks = load_and_split(sample_txt_file)
        for chunk in chunks:
            assert "source" in chunk.metadata
            # TextLoader sets source to the full path; check the basename is contained
            assert os.path.basename(sample_txt_file) in os.path.basename(
                chunk.metadata["source"]
            )

    def test_md_file_returns_chunks(self, sample_md_file):
        chunks = load_and_split(sample_md_file)
        assert len(chunks) > 0

    def test_csv_file_returns_chunks(self, sample_csv_file):
        chunks = load_and_split(sample_csv_file)
        assert isinstance(chunks, list)
        # CSV loader typically returns one doc per row
        assert len(chunks) >= 1

    def test_chunk_size_respected(self, sample_txt_file):
        """No chunk should exceed CHUNK_SIZE + CHUNK_OVERLAP characters."""
        chunks = load_and_split(sample_txt_file)
        max_allowed = cfg.CHUNK_SIZE + cfg.CHUNK_OVERLAP
        for chunk in chunks:
            assert len(chunk.page_content) <= max_allowed, (
                f"Chunk too long: {len(chunk.page_content)} > {max_allowed}"
            )

    def test_empty_loader_returns_empty_list(self, tmp_path):
        """If the loader raises, the function should return []."""
        bad_file = tmp_path / "broken.txt"
        bad_file.write_bytes(b"\xff\xfe")  # invalid UTF-8 — loader will error

        # Mock the loader to raise an Exception (not ValueError)
        with patch("backend.document_loader.TextLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = RuntimeError("decode error")
            result = load_and_split(str(bad_file))
        assert result == []

    def test_source_metadata_set_for_every_chunk(self, tmp_path):
        txt = tmp_path / "multi.txt"
        # Write enough text to produce at least 2 chunks
        txt.write_text(("word " * 200) + "\n" + ("word " * 200), encoding="utf-8")
        chunks = load_and_split(str(txt))
        for chunk in chunks:
            # source is set (either full path from loader, or basename via setdefault)
            assert "source" in chunk.metadata
            assert chunk.metadata["source"]  # non-empty


# ─────────────────────────────────────────────────────────────────────────────
# get_document_preview
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDocumentPreview:
    def test_txt_returns_string(self, sample_txt_file):
        result = get_document_preview(sample_txt_file)
        assert isinstance(result, str)

    def test_txt_preview_non_empty(self, sample_txt_file):
        result = get_document_preview(sample_txt_file)
        assert len(result) > 0

    def test_char_limit_is_respected(self, sample_txt_file):
        limit = 50
        result = get_document_preview(sample_txt_file, char_limit=limit)
        assert len(result) <= limit

    def test_full_preview_within_char_limit_for_short_doc(self, tmp_path):
        short = tmp_path / "short.txt"
        short.write_text("Hello world", encoding="utf-8")
        result = get_document_preview(str(short), char_limit=2000)
        assert "Hello world" in result

    def test_md_preview_returns_content(self, sample_md_file):
        result = get_document_preview(sample_md_file)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unsupported_extension_returns_message(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"fake excel")
        result = get_document_preview(str(f))
        assert "not available" in result.lower() or "xlsx" in result.lower()

    def test_loader_error_returns_error_message(self, tmp_path):
        f = tmp_path / "corrupt.txt"
        f.write_bytes(b"\x80\x81\x82")
        with patch("backend.document_loader.TextLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = Exception("decode error")
            result = get_document_preview(str(f))
        assert "error" in result.lower()
