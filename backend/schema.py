from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with content, page info, and metadata."""
    chunk_id: str = Field(..., description="Unique ID for this chunk")
    content: str = Field(..., description="The textual content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata (source, page, file_type, etc.)")
    score: Optional[float] = Field(default=None, description="Retrieval or reranking relevance score")


class Citation(BaseModel):
    """Citation linking an answer snippet to a specific document and page."""
    source: str = Field(..., description="Document filename")
    page: Optional[int] = Field(default=None, description="Page or slide number")
    snippet: str = Field(..., description="Excerpt from the source document")
    score: Optional[float] = Field(default=None, description="Relevance score")


class AgentState(TypedDict, total=False):
    """State passed between nodes in the LangGraph workflow."""
    # Inputs
    question: str
    original_question: str
    collection_id: str
    
    # Retrieval & Reranking
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    
    # Grading & Decision loop
    is_relevant: bool
    retry_count: int
    web_fallback: bool
    
    # Output
    answer: str
    citations: List[Dict[str, Any]]
    error: Optional[str]


# API Request & Response Models

class QuestionRequest(BaseModel):
    """Request model for the chat endpoint."""
    question: str = Field(..., description="User question")
    collection_id: Optional[str] = Field(default=None, description="Document/Session collection ID")
    file_info: Optional[Dict[str, Any]] = Field(default=None, description="Legacy fallback file info")


class UploadResponse(BaseModel):
    """Response model for the file upload endpoint."""
    status: str
    message: str
    filename: str
    collection_id: str
    num_pages: int
    num_chunks: int
    content_preview: str
    content: Optional[str] = Field(default=None, description="Preview content for backwards compatibility")


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    status: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    collection_id: Optional[str] = None


class DocumentInfo(BaseModel):
    """Metadata for an indexed document in the vector store."""
    collection_id: str
    filename: str
    file_type: str
    num_chunks: int
    created_at: float