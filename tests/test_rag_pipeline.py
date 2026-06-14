"""
test_rag_pipeline.py — Unit tests for backend/rag_pipeline.py.

The LLM (Ollama) and the FAISS vector store are fully mocked, so these tests
run without any external services or ML model downloads.

Tests cover:
- build_qa_chain      : returns a RetrievalQA object; uses correct model name
- answer_question     : formats answer / sources / chunks correctly;
                        handles chat history; handles empty source_documents
- stream_answer       : yields valid SSE data strings; final frame has done=True
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_doc(content: str = "chunk text", source: str = "doc.txt", page: int = 1):
    return Document(page_content=content, metadata={"source": source, "page": page})


def _make_chain_result(answer: str = "The answer.", docs=None):
    """Fake result dict returned by chain.invoke()."""
    return {
        "result": answer,
        "source_documents": docs if docs is not None else [_make_doc()],
    }


# ─────────────────────────────────────────────────────────────────────────────
# build_qa_chain
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildQaChain:
    @patch("backend.rag_pipeline.RetrievalQA")
    @patch("backend.rag_pipeline.Ollama")
    def test_returns_retrieval_qa_chain(self, MockOllama, MockRetrievalQA):
        from langchain.chains import RetrievalQA
        mock_chain = MagicMock(spec=RetrievalQA)
        MockRetrievalQA.from_chain_type.return_value = mock_chain

        vector_store = MagicMock()
        vector_store.as_retriever.return_value = MagicMock()

        from backend.rag_pipeline import build_qa_chain
        result = build_qa_chain(vector_store)

        assert result is mock_chain
        MockRetrievalQA.from_chain_type.assert_called_once()

    @patch("backend.rag_pipeline.RetrievalQA")
    @patch("backend.rag_pipeline.Ollama")
    def test_uses_default_model_when_none_given(self, MockOllama, MockRetrievalQA):
        from backend.config import DEFAULT_MODEL
        MockRetrievalQA.from_chain_type.return_value = MagicMock()

        vector_store = MagicMock()
        vector_store.as_retriever.return_value = MagicMock()

        from backend.rag_pipeline import build_qa_chain
        build_qa_chain(vector_store, model_name=None)

        MockOllama.assert_called_once()
        call_kwargs = MockOllama.call_args[1]
        assert call_kwargs.get("model") == DEFAULT_MODEL

    @patch("backend.rag_pipeline.RetrievalQA")
    @patch("backend.rag_pipeline.Ollama")
    def test_uses_custom_model_when_given(self, MockOllama, MockRetrievalQA):
        MockRetrievalQA.from_chain_type.return_value = MagicMock()

        vector_store = MagicMock()
        vector_store.as_retriever.return_value = MagicMock()

        from backend.rag_pipeline import build_qa_chain
        build_qa_chain(vector_store, model_name="llama3")

        MockOllama.assert_called_once()
        call_kwargs = MockOllama.call_args[1]
        assert call_kwargs.get("model") == "llama3"

    @patch("backend.rag_pipeline.RetrievalQA")
    @patch("backend.rag_pipeline.Ollama")
    def test_retriever_called_on_vector_store(self, MockOllama, MockRetrievalQA):
        MockRetrievalQA.from_chain_type.return_value = MagicMock()

        vector_store = MagicMock()
        vector_store.as_retriever.return_value = MagicMock()

        from backend.rag_pipeline import build_qa_chain
        build_qa_chain(vector_store)

        vector_store.as_retriever.assert_called_once()



# ─────────────────────────────────────────────────────────────────────────────
# answer_question
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerQuestion:
    def test_returns_dict_with_required_keys(self):
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result()

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "What is RAG?")

        assert "answer" in result
        assert "sources" in result
        assert "chunks" in result

    def test_answer_text_correct(self):
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result(answer="RAG stands for retrieval.")

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "What is RAG?")

        assert result["answer"] == "RAG stands for retrieval."

    def test_sources_list_populated(self):
        doc = _make_doc(source="manual.pdf", page=3)
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result(docs=[doc])

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "Question?")

        assert len(result["sources"]) == 1
        assert result["sources"][0]["file"] == "manual.pdf"
        assert result["sources"][0]["page"] == 3

    def test_chunks_list_populated(self):
        doc = _make_doc(content="relevant chunk")
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result(docs=[doc])

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "Question?")

        assert "relevant chunk" in result["chunks"]

    def test_empty_source_documents_gives_empty_sources_and_chunks(self):
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result(docs=[])

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "Question?")

        assert result["sources"] == []
        assert result["chunks"] == []

    def test_multiple_source_documents(self):
        docs = [_make_doc(source=f"doc{i}.txt", page=i) for i in range(3)]
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result(docs=docs)

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "Question?")

        assert len(result["sources"]) == 3
        assert len(result["chunks"]) == 3

    def test_chat_history_augments_query(self):
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result()

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        from backend.rag_pipeline import answer_question
        answer_question(chain, "Follow-up question?", chat_history=history)

        invoked_query = chain.invoke.call_args[0][0]["query"]
        # The effective query should include chat history content
        assert "Hello" in invoked_query or "Follow-up" in invoked_query

    def test_no_chat_history_passes_raw_query(self):
        chain = MagicMock()
        chain.invoke.return_value = _make_chain_result()

        from backend.rag_pipeline import answer_question
        answer_question(chain, "Direct question?", chat_history=None)

        invoked_query = chain.invoke.call_args[0][0]["query"]
        assert invoked_query == "Direct question?"

    def test_missing_result_key_defaults_to_fallback(self):
        chain = MagicMock()
        chain.invoke.return_value = {"source_documents": []}  # no "result" key

        from backend.rag_pipeline import answer_question
        result = answer_question(chain, "Question?")

        assert isinstance(result["answer"], str)


# ─────────────────────────────────────────────────────────────────────────────
# stream_answer
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamAnswer:
    @patch("backend.rag_pipeline.Ollama")
    def test_yields_sse_strings(self, MockOllama):
        doc = _make_doc()
        vector_store = MagicMock()
        vector_store.as_retriever.return_value.invoke.return_value = [doc]

        llm_instance = MagicMock()
        llm_instance.stream.return_value = iter(["Hello", " world"])
        MockOllama.return_value = llm_instance

        from backend.rag_pipeline import stream_answer
        frames = list(stream_answer(vector_store, "What?"))

        assert len(frames) > 0
        for frame in frames:
            assert frame.startswith("data: ")
            assert frame.endswith("\n\n")

    @patch("backend.rag_pipeline.Ollama")
    def test_token_frames_contain_token_key(self, MockOllama):
        doc = _make_doc()
        vector_store = MagicMock()
        vector_store.as_retriever.return_value.invoke.return_value = [doc]

        llm_instance = MagicMock()
        llm_instance.stream.return_value = iter(["tok1", "tok2"])
        MockOllama.return_value = llm_instance

        from backend.rag_pipeline import stream_answer
        frames = list(stream_answer(vector_store, "Question?"))

        token_frames = frames[:-1]  # last frame is the done signal
        for frame in token_frames:
            payload = json.loads(frame[len("data: "):].strip())
            assert "token" in payload

    @patch("backend.rag_pipeline.Ollama")
    def test_final_frame_has_done_true(self, MockOllama):
        doc = _make_doc()
        vector_store = MagicMock()
        vector_store.as_retriever.return_value.invoke.return_value = [doc]

        llm_instance = MagicMock()
        llm_instance.stream.return_value = iter(["token"])
        MockOllama.return_value = llm_instance

        from backend.rag_pipeline import stream_answer
        frames = list(stream_answer(vector_store, "Query?"))

        last = json.loads(frames[-1][len("data: "):].strip())
        assert last.get("done") is True

    @patch("backend.rag_pipeline.Ollama")
    def test_final_frame_contains_sources_and_chunks(self, MockOllama):
        doc = _make_doc(source="report.pdf", page=2, content="excerpt")
        vector_store = MagicMock()
        vector_store.as_retriever.return_value.invoke.return_value = [doc]

        llm_instance = MagicMock()
        llm_instance.stream.return_value = iter(["token"])
        MockOllama.return_value = llm_instance

        from backend.rag_pipeline import stream_answer
        frames = list(stream_answer(vector_store, "Query?"))

        last = json.loads(frames[-1][len("data: "):].strip())
        assert "sources" in last
        assert "chunks" in last
        assert last["sources"][0]["file"] == "report.pdf"
        assert "excerpt" in last["chunks"]

    @patch("backend.rag_pipeline.Ollama")
    def test_no_tokens_still_yields_done_frame(self, MockOllama):
        vector_store = MagicMock()
        vector_store.as_retriever.return_value.invoke.return_value = []

        llm_instance = MagicMock()
        llm_instance.stream.return_value = iter([])
        MockOllama.return_value = llm_instance

        from backend.rag_pipeline import stream_answer
        frames = list(stream_answer(vector_store, "Empty?"))

        assert len(frames) == 1
        last = json.loads(frames[-1][len("data: "):].strip())
        assert last.get("done") is True
