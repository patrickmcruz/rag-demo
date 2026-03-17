"""Query endpoint."""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.deps import get_ready_rag_app
from src.api.models import QueryRequest, QueryResponse, SourceDocument
from src.app import RAGApplication

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_app: Annotated[RAGApplication, Depends(get_ready_rag_app)],
) -> QueryResponse:
    """Run a single synchronous RAG query and return the answer with sources."""
    rag_app.apply_overrides(
        top_k_documents=request.top_k,
        temperature=request.temperature,
    )

    # RAGApplication.query() is synchronous (Ollama + ChromaDB).
    # run_in_executor prevents blocking the async event loop.
    loop = asyncio.get_event_loop()
    rag_response = await loop.run_in_executor(
        None,
        lambda: rag_app.query(
            request.question,
            return_sources=request.return_sources,
            verbose=False,
        ),
    )

    sources = [
        SourceDocument(
            source=doc.metadata.get("source", "Unknown"),
            content_preview=doc.page_content[:200],
            page=doc.metadata.get("page"),
        )
        for doc in rag_response.sources
    ]

    return QueryResponse(
        answer=rag_response.answer,
        sources=sources,
        query=rag_response.query,
        response_time=rag_response.response_time,
        model_name=rag_response.model_name,
    )
