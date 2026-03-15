"""
app.py — Streamlit frontend for the RAG Document Q&A System.

Features:
    • Chat-style interface with conversation history
    • Model selector dropdown (Mistral, Llama3, Gemma)
    • File upload for PDF, TXT, DOCX, CSV, MD
    • Document preview in sidebar
    • Chunk visualization (retrieved passages)
    • Streaming responses via SSE
    • API key authentication
    • Response timing
"""

import json
import time
from typing import Generator

import requests
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "rag-secret-key-2024"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Session state initialisation ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = DEFAULT_API_KEY
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "mistral"


def _headers() -> dict:
    """Return standard request headers with API key."""
    return {"X-API-Key": st.session_state.api_key}


def _stream_tokens(question: str, model: str, history: list) -> Generator[str, None, dict]:
    """Call the /ask/stream SSE endpoint and yield tokens.

    Returns the final metadata dict (sources, chunks) via generator return.
    """
    payload = {"question": question, "model": model}
    if history:
        payload["chat_history"] = history

    metadata = {"sources": [], "chunks": []}
    try:
        with requests.post(
            f"{API_BASE_URL}/ask/stream",
            json=payload,
            headers=_headers(),
            stream=True,
            timeout=180,
        ) as resp:
            if resp.status_code != 200:
                yield f"❌ Error: {resp.text}"
                return metadata

            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "token" in data:
                        yield data["token"]
                    if data.get("done"):
                        metadata["sources"] = data.get("sources", [])
                        metadata["chunks"] = data.get("chunks", [])
    except requests.exceptions.ConnectionError:
        yield "❌ Cannot connect to the backend."
    except Exception as exc:
        yield f"❌ Error: {exc}"

    return metadata


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Document Q&A")
    st.markdown("---")

    # ── API Key ──────────────────────────────────────────────────────────────
    st.subheader("🔑 API Key")
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="Required for all API calls",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    st.markdown("---")

    # ── Model selector ───────────────────────────────────────────────────────
    st.subheader("🤖 LLM Model")
    available_models = ["mistral", "llama3", "gemma"]
    try:
        models_resp = requests.get(
            f"{API_BASE_URL}/models", headers=_headers(), timeout=5
        )
        if models_resp.status_code == 200:
            available_models = models_resp.json().get("models", available_models)
    except Exception:
        pass

    st.session_state.selected_model = st.selectbox(
        "Select model",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models
        else 0,
    )

    st.markdown("---")

    # ── File uploader ────────────────────────────────────────────────────────
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx", "csv", "md"],
        help="Supported: PDF, TXT, DOCX, CSV, MD",
    )

    if st.button("📤 Upload & Index", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Uploading and indexing…"):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/upload",
                        files=files,
                        headers=_headers(),
                        timeout=120,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"✅ **{data['filename']}** indexed "
                            f"({data['chunks']} chunks)"
                        )
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"❌ Upload failed: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to backend. "
                        "Is FastAPI running on http://localhost:8000?"
                    )
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")
        else:
            st.warning("⚠️ Please select a file first.")

    st.markdown("---")

    # ── Indexed documents with preview ───────────────────────────────────────
    st.subheader("📂 Indexed Documents")
    try:
        doc_response = requests.get(
            f"{API_BASE_URL}/documents", headers=_headers(), timeout=10
        )
        if doc_response.status_code == 200:
            documents = doc_response.json().get("documents", [])
            if documents:
                for doc_name in documents:
                    with st.expander(f"📄 {doc_name}"):
                        try:
                            preview_resp = requests.get(
                                f"{API_BASE_URL}/documents/{doc_name}/preview",
                                headers=_headers(),
                                timeout=10,
                            )
                            if preview_resp.status_code == 200:
                                preview = preview_resp.json().get("preview", "")
                                st.text(preview[:1000] + ("…" if len(preview) > 1000 else ""))
                            else:
                                st.caption("Preview unavailable.")
                        except Exception:
                            st.caption("Preview unavailable.")
            else:
                st.info("No documents indexed yet.")
        else:
            st.warning("Could not fetch document list.")
    except requests.exceptions.ConnectionError:
        st.info("Backend not reachable — start the FastAPI server first.")
    except Exception as exc:
        st.warning(f"Error listing documents: {exc}")

    st.markdown("---")

    # ── Clear all data ───────────────────────────────────────────────────────
    st.subheader("🗑️ Manage Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Data", use_container_width=True):
            try:
                clear_resp = requests.delete(
                    f"{API_BASE_URL}/clear", headers=_headers(), timeout=10
                )
                if clear_resp.status_code == 200:
                    st.session_state.chat_history = []
                    st.success("✅ All data cleared.")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear data.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend.")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ── Main area — Chat Interface ───────────────────────────────────────────────
st.header("🔍 Ask Questions About Your Documents")
st.markdown(
    "Upload documents via the sidebar, then ask questions below. "
    "The system uses **RAG** to find relevant passages and generate answers. "
    f"Model: **{st.session_state.selected_model}**"
)

# Check if any documents exist
has_documents = False
try:
    check_response = requests.get(
        f"{API_BASE_URL}/documents", headers=_headers(), timeout=5
    )
    if check_response.status_code == 200:
        has_documents = bool(check_response.json().get("documents"))
except Exception:
    pass

if not has_documents:
    st.info(
        "📂 **No documents indexed yet.** Upload a file using the sidebar to get started."
    )

# ── Display chat history ─────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources and chunks for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src.get('file', 'unknown')}** "
                        f"(page {src.get('page', 'N/A')})"
                    )
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("🔍 Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500])
                    st.markdown("---")
        if msg["role"] == "assistant" and msg.get("elapsed_ms"):
            st.caption(f"⏱️ {msg['elapsed_ms']:.0f} ms")

# ── Chat input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your documents…")

if question:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build chat history for API (exclude metadata like sources/chunks)
    api_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history[:-1]
    ]

    # Get answer
    with st.chat_message("assistant"):
        start_time = time.time()
        try:
            # Try streaming first
            full_answer = ""
            metadata = {"sources": [], "chunks": []}
            placeholder = st.empty()

            payload = {
                "question": question,
                "model": st.session_state.selected_model,
            }
            if api_history:
                payload["chat_history"] = api_history

            try:
                with requests.post(
                    f"{API_BASE_URL}/ask/stream",
                    json=payload,
                    headers=_headers(),
                    stream=True,
                    timeout=180,
                ) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines(decode_unicode=True):
                            if line and line.startswith("data: "):
                                data = json.loads(line[6:])
                                if "token" in data:
                                    full_answer += data["token"]
                                    placeholder.markdown(full_answer + "▌")
                                if data.get("done"):
                                    metadata["sources"] = data.get("sources", [])
                                    metadata["chunks"] = data.get("chunks", [])

                        placeholder.markdown(full_answer)
                    else:
                        # Fall back to non-streaming
                        raise ConnectionError("Stream failed")
            except Exception:
                # Fallback: use non-streaming /ask endpoint
                ask_resp = requests.post(
                    f"{API_BASE_URL}/ask",
                    json=payload,
                    headers=_headers(),
                    timeout=120,
                )
                if ask_resp.status_code == 200:
                    result = ask_resp.json()
                    full_answer = result["answer"]
                    metadata["sources"] = result.get("sources", [])
                    metadata["chunks"] = result.get("chunks", [])
                    placeholder.markdown(full_answer)
                else:
                    detail = ask_resp.json().get("detail", ask_resp.text)
                    full_answer = f"❌ {detail}"
                    placeholder.error(full_answer)

            elapsed_ms = (time.time() - start_time) * 1000

            # Show sources
            if metadata["sources"]:
                with st.expander("📎 Sources", expanded=False):
                    for src in metadata["sources"]:
                        st.markdown(
                            f"- **{src.get('file', 'unknown')}** "
                            f"(page {src.get('page', 'N/A')})"
                        )

            # Show retrieved chunks
            if metadata["chunks"]:
                with st.expander("🔍 Retrieved Chunks", expanded=False):
                    for i, chunk in enumerate(metadata["chunks"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(chunk[:500])
                        st.markdown("---")

            st.caption(f"⏱️ {elapsed_ms:.0f} ms")

            # Save to history
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "sources": metadata["sources"],
                    "chunks": metadata["chunks"],
                    "elapsed_ms": elapsed_ms,
                }
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to backend. "
                "Is FastAPI running on http://localhost:8000?"
            )
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
