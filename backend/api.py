import os
import time
import uuid
import shutil
import tempfile
import traceback
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schema import QuestionRequest, UploadResponse, ChatResponse, Citation
from .parsers import DocumentParser
from .vector_store import VectorStoreManager
from .agent import AgenticRAG
from . import config

app = FastAPI(
    title="Document Intelligence & Agentic RAG API",
    description="Enterprise-grade Document QA API with LangGraph, Qdrant Hybrid Search, and FlashRank",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton service instances to prevent lock conflicts with Uvicorn reloader
_parser: Optional[DocumentParser] = None
_vector_store: Optional[VectorStoreManager] = None
_agent: Optional[AgenticRAG] = None


def get_parser() -> DocumentParser:
    global _parser
    if _parser is None:
        _parser = DocumentParser()
    return _parser


def get_vector_store() -> VectorStoreManager:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store


def get_agent() -> AgenticRAG:
    global _agent
    if _agent is None:
        _agent = AgenticRAG(vector_store=get_vector_store())
    return _agent


# Track uploaded documents metadata
DOCUMENT_REGISTRY = {}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests, parses, chunks, and indexes a document into Qdrant vector store.
    Supports PDF, DOCX, PPTX, XLSX, and TXT natively.
    """
    try:
        filename = file.filename or "uploaded_document"
        file_ext = os.path.splitext(filename)[1].lower()
        supported = [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".csv"]

        if file_ext not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{file_ext}'. Supported: {', '.join(supported)}"
            )

        # Unique collection ID for session isolation
        clean_name = re_clean_name(filename)
        collection_id = f"{clean_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        file_path = os.path.join(config.UPLOAD_DIR, f"{collection_id}{file_ext}")

        # Stream file to disk in chunks to avoid memory spikes
        chunk_size = 1024 * 1024  # 1MB
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)

        print(f"Uploaded file saved to {file_path}. Starting layout-aware parsing...")

        # Parse & Chunk with page metadata
        chunks, total_pages, preview = get_parser().parse_and_chunk(file_path)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text or tabular content could be extracted from this document."
            )

        # Index chunks into Qdrant
        num_indexed = get_vector_store().index_chunks(collection_id=collection_id, chunks=chunks)

        # Store metadata
        DOCUMENT_REGISTRY[collection_id] = {
            "collection_id": collection_id,
            "filename": filename,
            "file_path": file_path,
            "total_pages": total_pages,
            "num_chunks": num_indexed,
            "created_at": time.time()
        }

        return UploadResponse(
            status="success",
            message="Document successfully parsed and indexed into vector database.",
            filename=filename,
            collection_id=collection_id,
            num_pages=total_pages,
            num_chunks=num_indexed,
            content_preview=preview,
            content=preview
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload endpoint: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_document(request: QuestionRequest):
    """
    Answers questions using the Corrective RAG (CRAG) LangGraph agent.
    Includes document citations with exact page numbers.
    """
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        collection_id = request.collection_id

        # Fallback to the most recent collection if not provided directly
        if not collection_id:
            collections = get_vector_store().list_collections()
            if not collections:
                raise HTTPException(
                    status_code=400,
                    detail="No document has been uploaded or indexed yet. Please upload a document first."
                )
            collection_id = collections[-1]

        print(f"Executing Agentic RAG for question: '{question}' on collection: '{collection_id}'")
        result = get_agent().run(question=question, collection_id=collection_id)

        citations_list = [
            Citation(
                source=c.get("source", "Document"),
                page=c.get("page"),
                snippet=c.get("snippet", ""),
                score=c.get("score")
            )
            for c in result.get("citations", [])
        ]

        return ChatResponse(
            status="success",
            answer=result.get("answer", "No answer could be generated."),
            citations=citations_list,
            collection_id=collection_id
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: QuestionRequest):
    """
    Streaming SSE endpoint — pushes answer tokens as they are generated.
    Client receives Server-Sent Events with {token} chunks, followed by
    a final {citations, done: true} event.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    collection_id = request.collection_id
    if not collection_id:
        collections = get_vector_store().list_collections()
        if not collections:
            raise HTTPException(
                status_code=400,
                detail="No document has been uploaded or indexed yet. Please upload a document first."
            )
        collection_id = collections[-1]

    print(f"[Stream] Starting streaming RAG for: '{question}' on '{collection_id}'")

    def event_generator():
        try:
            yield from get_agent().stream_answer(
                question=question,
                collection_id=collection_id
            )
        except Exception as e:
            import json
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/documents")
async def get_indexed_documents():
    """Lists all active document collections in the vector store."""
    collections = get_vector_store().list_collections()
    docs = []
    for col in collections:
        meta = DOCUMENT_REGISTRY.get(col, {
            "collection_id": col,
            "filename": col,
            "created_at": time.time()
        })
        docs.append(meta)
    return {"collections": docs}


@app.delete("/api/documents/{collection_id}")
async def delete_document(collection_id: str):
    """Deletes an indexed document collection from Qdrant and disk."""
    try:
        get_vector_store().delete_collection(collection_id)
        if collection_id in DOCUMENT_REGISTRY:
            meta = DOCUMENT_REGISTRY.pop(collection_id)
            if os.path.exists(meta.get("file_path", "")):
                os.remove(meta["file_path"])
        return {"status": "success", "message": f"Collection '{collection_id}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def re_clean_name(name: str) -> str:
    """Sanitizes filename for safe collection naming."""
    base = os.path.splitext(name)[0]
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)[:20]


def run_api_server(host="0.0.0.0", port=8000, reload=False):
    """Launches the Uvicorn ASGI server."""
    import uvicorn
    frontend_build = os.path.join(config.BASE_DIR, "frontend", "build")
    if os.path.exists(frontend_build):
        app.mount("/", StaticFiles(directory=frontend_build, html=True), name="frontend")
    uvicorn.run("backend.api:app", host=host, port=port, reload=reload)