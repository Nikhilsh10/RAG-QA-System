"""
main.py — FastAPI application entry point for the RAG Document Q&A System.

Endpoints:
    POST   /upload                    Upload and index a document
    POST   /ask                       Ask a question (with optional model + history)
    POST   /ask/stream                Stream answer via SSE
    GET    /documents                 List all uploaded documents
    GET    /documents/{filename}/preview  Preview document text
    GET    /models                    List available LLM models
    DELETE /clear                     Remove all documents and the FAISS index
    GET    /health                    Health-check (no auth required)
"""

import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import (
    ALLOWED_EXTENSIONS,
    API_KEY,
    AVAILABLE_MODELS,
    PREVIEW_CHAR_LIMIT,
    UPLOAD_DIR,
    VECTOR_DB_DIR,
)
from backend.document_loader import get_document_preview, load_and_split
from backend.rag_pipeline import answer_question, build_qa_chain, stream_answer
from backend.vector_store import add_to_vector_store, load_vector_store

# ── Pydantic models ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str


class QuestionRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str
    model: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None


class UploadResponse(BaseModel):
    """Response body for the /upload endpoint."""

    message: str
    filename: str
    chunks: int


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    answer: str
    sources: list
    chunks: list


class DocumentsResponse(BaseModel):
    """Response body for the /documents endpoint."""

    documents: List[str]


class PreviewResponse(BaseModel):
    """Response body for the document preview endpoint."""

    filename: str
    preview: str


class ModelsResponse(BaseModel):
    """Response body for the /models endpoint."""

    models: List[str]


class MessageResponse(BaseModel):
    """Generic response with a message field."""

    message: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Q&A System",
    description="Upload documents and ask questions powered by RAG.",
    version="2.0.0",
)

# CORS — allow all origins for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Authentication Middleware ────────────────────────────────────────


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Check X-API-Key header on all endpoints except /health and /docs."""
    skip_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    if request.url.path not in skip_paths:
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )
    response = await call_next(request)
    return response


# ── Startup event ────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    """Create required directories on application startup."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a document file, save it, and index its contents.

    Supported formats: PDF, TXT, DOCX, CSV, MD.

    Args:
        file: The uploaded file (multipart form data).

    Returns:
        Confirmation message, filename, and number of chunks indexed.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{extension}'. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    chunks = load_and_split(file_path)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any content from the uploaded file.",
        )

    add_to_vector_store(chunks)

    return UploadResponse(
        message="File uploaded and indexed",
        filename=file.filename,
        chunks=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(body: QuestionRequest) -> AskResponse:
    """Answer a question using the indexed documents.

    Optionally accepts a ``model`` name and ``chat_history`` for
    conversational context.

    Args:
        body: JSON body with question, optional model, optional chat_history.

    Returns:
        The LLM's answer, source references, and retrieved chunk texts.
    """
    vector_store = load_vector_store()
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    chain = build_qa_chain(vector_store, model_name=body.model)

    history = None
    if body.chat_history:
        history = [msg.model_dump() for msg in body.chat_history]

    result = answer_question(chain, body.question, chat_history=history)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks=result["chunks"],
    )


@app.post("/ask/stream")
async def ask_question_stream(body: QuestionRequest):
    """Stream an answer token-by-token via Server-Sent Events.

    Args:
        body: JSON body with question, optional model, optional chat_history.

    Returns:
        A ``text/event-stream`` streaming response.
    """
    vector_store = load_vector_store()
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    history = None
    if body.chat_history:
        history = [msg.model_dump() for msg in body.chat_history]

    return StreamingResponse(
        stream_answer(
            vector_store,
            body.question,
            model_name=body.model,
            chat_history=history,
        ),
        media_type="text/event-stream",
    )


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    """List all files currently stored in the upload directory.

    Returns:
        A list of filenames.
    """
    if not os.path.exists(UPLOAD_DIR):
        return DocumentsResponse(documents=[])

    files = [
        f
        for f in os.listdir(UPLOAD_DIR)
        if os.path.isfile(os.path.join(UPLOAD_DIR, f))
    ]
    return DocumentsResponse(documents=sorted(files))


@app.get("/documents/{filename}/preview", response_model=PreviewResponse)
async def preview_document(filename: str) -> PreviewResponse:
    """Return a plain-text preview of an uploaded document.

    Args:
        filename: Name of the file in the upload directory.

    Returns:
        Filename and the first ~2000 characters of its content.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    preview_text = get_document_preview(file_path, char_limit=PREVIEW_CHAR_LIMIT)
    return PreviewResponse(filename=filename, preview=preview_text)


@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Return the list of available LLM models.

    Returns:
        A list of model name strings.
    """
    return ModelsResponse(models=AVAILABLE_MODELS)


@app.delete("/clear", response_model=MessageResponse)
async def clear_all_data() -> MessageResponse:
    """Delete all uploaded documents and the FAISS index.

    Returns:
        A confirmation message.
    """
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    return MessageResponse(message="All data cleared")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple health-check endpoint (no authentication required).

    Returns:
        ``{"status": "ok"}``
    """
    return HealthResponse(status="ok")


# ── Run with uvicorn ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
