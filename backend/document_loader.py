"""
document_loader.py — Load and split PDF / TXT / DOCX / CSV / MD documents
into LangChain Document chunks with metadata.
"""

import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split(file_path: str) -> List[Document]:
    """Load a document file and split it into chunks.

    Supported formats: PDF, TXT, DOCX, CSV, MD.
    The file type is auto-detected from its extension.  Each returned
    ``Document`` carries metadata with the source filename and, for PDFs,
    the page number.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        A list of ``Document`` chunks ready for embedding.

    Raises:
        ValueError: If the file extension is not supported.
    """
    extension = os.path.splitext(file_path)[1].lower()

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported file type: '{extension}'. "
                f"Supported: .pdf, .txt, .docx, .csv, .md"
            )

        documents: List[Document] = loader.load()
    except ValueError:
        raise
    except Exception as exc:
        print(f"[WARNING] Failed to load '{file_path}': {exc}")
        return []

    # Attach source filename to every document's metadata
    source_filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata.setdefault("source", source_filename)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    try:
        chunks: List[Document] = text_splitter.split_documents(documents)
    except Exception as exc:
        print(f"[WARNING] Failed to split '{file_path}': {exc}")
        return []

    return chunks


def get_document_preview(file_path: str, char_limit: int = 2000) -> str:
    """Return the first ``char_limit`` characters of a document's text.

    Args:
        file_path:  Path to the document file.
        char_limit: Maximum number of characters to return.

    Returns:
        A truncated plain-text preview of the document.
    """
    extension = os.path.splitext(file_path)[1].lower()

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        else:
            return f"[Preview not available for {extension} files]"

        docs = loader.load()
        full_text = "\n".join(doc.page_content for doc in docs)
        return full_text[:char_limit]
    except Exception as exc:
        return f"[Error loading preview: {exc}]"
