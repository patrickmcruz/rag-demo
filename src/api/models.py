"""Pydantic v2 request and response schemas for the RAG API."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    return_sources: bool = True
    language: Literal["pt", "en"] = "pt"
    top_k: Optional[int] = Field(None, gt=0, le=50)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class IngestRequest(BaseModel):
    data_dir: Optional[str] = None
    vectorstore_dir: Optional[str] = None
    file_types: Optional[List[str]] = None
    chunk_size: Optional[int] = Field(None, gt=0, le=10000)
    chunk_overlap: Optional[int] = Field(None, ge=0)
    embedding_model: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: Literal["pt", "en"] = "pt"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    chain_ready: bool
    vectorstore_exists: bool


class SourceDocument(BaseModel):
    source: str
    content_preview: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    response_time: float
    model_name: str


class IngestAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    message: str = "Ingestion job queued. Poll /ingest/{job_id} for status."


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    docs_loaded: Optional[int] = None
    chunks_created: Optional[int] = None


class InfoResponse(BaseModel):
    model: str
    embedding_model: str
    data_dir: str
    vectorstore_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k_documents: int
    temperature: float
    use_gpu: bool
    gpu_device: int
    vectorstore_exists: bool
    data_dir_exists: bool
    data_file_counts: Dict[str, int]
