"""
rag_pipeline.py — Core Retrieval-Augmented Generation logic.

Supports:
- Multi-model selection (any Ollama model)
- Standard QA with source + chunk text retrieval
- Streaming token-by-token answers via Ollama
- Conversational QA with chat history
"""

import json
from typing import Any, Dict, Generator, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from backend.config import DEFAULT_MODEL, TEMPERATURE, TOP_K_RESULTS

# ── Custom prompt ────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """You are a helpful assistant. Use only the context below to answer \
the question. If the answer is not in the context, say \
"I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

_QA_PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ── Conversational prompt (includes chat history) ────────────────────────────
_CONVERSATIONAL_TEMPLATE = """You are a helpful assistant. Use only the context below to answer \
the question. Consider the chat history for follow-up context. If the \
answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

_CONVERSATIONAL_PROMPT = PromptTemplate(
    template=_CONVERSATIONAL_TEMPLATE,
    input_variables=["context", "chat_history", "question"],
)


def build_qa_chain(
    vector_store: FAISS,
    model_name: Optional[str] = None,
) -> RetrievalQA:
    """Construct a RetrievalQA chain from a FAISS vector store.

    Args:
        vector_store: An initialised ``FAISS`` vector store.
        model_name:   Ollama model name (defaults to ``config.DEFAULT_MODEL``).

    Returns:
        A ready-to-invoke ``RetrievalQA`` chain.
    """
    llm = Ollama(
        model=model_name or DEFAULT_MODEL,
        temperature=TEMPERATURE,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _QA_PROMPT},
    )

    return qa_chain


def answer_question(
    chain: RetrievalQA,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run a question through the QA chain and format the result.

    If ``chat_history`` is provided the history is prepended to the query
    so the model can handle follow-up questions.

    Args:
        chain:        A ``RetrievalQA`` chain.
        query:        The user's question.
        chat_history: Optional list of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        A dict with ``answer``, ``sources``, and ``chunks`` (retrieved text).
    """
    # Build augmented query when chat history is present
    effective_query = query
    if chat_history:
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
        )
        effective_query = (
            f"Chat History:\n{history_text}\n\nCurrent Question: {query}"
        )

    result = chain.invoke({"query": effective_query})

    answer: str = result.get("result", "No answer returned.")

    sources: List[Dict[str, Any]] = []
    chunks: List[str] = []
    for doc in result.get("source_documents", []):
        sources.append(
            {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
            }
        )
        chunks.append(doc.page_content)

    return {"answer": answer, "sources": sources, "chunks": chunks}


def stream_answer(
    vector_store: FAISS,
    query: str,
    model_name: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """Stream an answer token-by-token as Server-Sent Events data.

    Retrieves relevant documents first, builds the prompt, then streams
    tokens from Ollama.

    Args:
        vector_store: The FAISS vector store.
        query:        The user's question.
        model_name:   Ollama model to use.
        chat_history: Optional conversation history.

    Yields:
        JSON-encoded SSE data strings, each containing a ``token`` or
        ``sources``/``chunks`` at the end.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Build prompt
    history_str = ""
    if chat_history:
        history_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
        )

    if history_str:
        prompt = _CONVERSATIONAL_TEMPLATE.format(
            context=context, chat_history=history_str, question=query
        )
    else:
        prompt = _PROMPT_TEMPLATE.format(context=context, question=query)

    # Stream from Ollama
    llm = Ollama(
        model=model_name or DEFAULT_MODEL,
        temperature=TEMPERATURE,
    )

    for token in llm.stream(prompt):
        yield f"data: {json.dumps({'token': token})}\n\n"

    # Send sources and chunks at the end
    sources = [
        {
            "file": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0),
        }
        for doc in docs
    ]
    chunks_text = [doc.page_content for doc in docs]

    yield f"data: {json.dumps({'done': True, 'sources': sources, 'chunks': chunks_text})}\n\n"
