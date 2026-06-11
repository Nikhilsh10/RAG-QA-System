# RAG-Based Document Q&A System

A production-ready Retrieval-Augmented Generation system that lets you
upload documents (PDF, TXT, DOCX, CSV, MD), index them with FAISS vector
embeddings, and ask natural-language questions answered by a local LLM
via Ollama — with chat history, streaming, and multi-model support.

---

## ✨ Features--

| Feature | Description |
|---------|-------------|
| 📄 Multi-format Upload | PDF, TXT, DOCX, CSV, Markdown |
| 🤖 Multi-model Support | Switch between Mistral, Llama3, Gemma |
| 💬 Chat History | Conversational follow-up questions |
| ⚡ Streaming Responses | Token-by-token SSE streaming |
| 🔍 Chunk Visualization | View retrieved text passages |
| 👁️ Document Preview | Preview uploaded documents in sidebar |
| 🔑 API Key Auth | Protected endpoints with X-API-Key |
| 🐳 Docker Ready | Full docker-compose with Ollama |

---

## Architecture

```
┌──────────────┐    ┌────────────┐    ┌───────────┐    ┌───────────────┐
│  PDF / TXT   │───▶│  Document  │───▶│  Text     │───▶│  HuggingFace  │
│  DOCX / CSV  │    │  Loader    │    │  Chunker  │    │  Embedder     │
│  Markdown    │    └────────────┘    └───────────┘    └──────┬────────┘
└──────────────┘                                              │
                                                              ▼
┌──────────────┐    ┌────────────┐    ┌───────────┐    ┌──────────────┐
│  Streaming   │◀───│  Ollama    │◀───│ Retriever │◀───│  FAISS       │
│  Answer      │    │  LLM      │    │  (top-k)  │    │  Vector DB   │
└──────────────┘    └────────────┘    └───────────┘    └──────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Streamlit Chat UI — history, sources, chunks, model selector      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download from <https://ollama.com> and pull a model:

```bash
ollama pull mistral
# Optional extras:
ollama pull llama3
ollama pull gemma
```

### 3. Run the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at **http://localhost:8000**.

### 4. Run the Streamlit frontend

```bash
streamlit run frontend/app.py
```

The UI will open at **http://localhost:8501**.

---

## Docker Setup

```bash
docker-compose up --build
```

This starts three services:
- **Ollama** on port 11434
- **FastAPI** on port 8000
- **Streamlit** on port 8501

Then pull a model inside the container:
```bash
docker exec -it rag-ollama ollama pull mistral
```

---

## API Endpoints

| Method   | Path                            | Auth | Description                           |
|----------|---------------------------------|------|---------------------------------------|
| `POST`   | `/upload`                       | ✅   | Upload and index a document           |
| `POST`   | `/ask`                          | ✅   | Ask a question (with model/history)   |
| `POST`   | `/ask/stream`                   | ✅   | Stream answer via SSE                 |
| `GET`    | `/documents`                    | ✅   | List uploaded documents               |
| `GET`    | `/documents/{filename}/preview` | ✅   | Preview document text                 |
| `GET`    | `/models`                       | ✅   | List available LLM models             |
| `DELETE` | `/clear`                        | ✅   | Delete all documents and FAISS index  |
| `GET`    | `/health`                       | ❌   | Health check (no auth)                |

### Authentication

All endpoints (except `/health`) require:
```
X-API-Key: rag-secret-key-2024
```

Set via environment variable `RAG_API_KEY`.

---

## Tech Stack

| Component        | Technology                              |
|------------------|-----------------------------------------|
| Backend API      | FastAPI + Uvicorn                       |
| Frontend         | Streamlit (chat UI)                     |
| LLM              | Mistral / Llama3 / Gemma (via Ollama)   |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2  |
| Vector Store     | FAISS (faiss-cpu)                       |
| Orchestration    | LangChain                               |
| Document Parsing | PyPDF, Docx2txt, CSVLoader, TextLoader  |
| Containerisation | Docker + docker-compose                 |

---

## Project Structure

```
rag-qa-system/
├── backend/
│   ├── main.py               # FastAPI app (auth, streaming, preview)
│   ├── rag_pipeline.py       # RAG logic (multi-model, streaming, history)
│   ├── embeddings.py         # HuggingFace embedding model loader
│   ├── vector_store.py       # FAISS index manager
│   ├── document_loader.py    # Multi-format document parser
│   └── config.py             # Configuration and constants
├── frontend/
│   └── app.py                # Streamlit chat UI
├── data/
│   └── uploaded_docs/        # Uploaded documents
├── vector_db/                # FAISS index storage
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

MIT
