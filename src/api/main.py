"""FastAPI application factory with lifespan management."""

import os
import warnings
from contextlib import asynccontextmanager
from typing import AsyncIterator

warnings.filterwarnings(
    "ignore",
    message=r".*RequestsDependencyWarning.*|urllib3 .* or chardet .* doesn't match a supported version!",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import register_exception_handlers
from src.api.routers import chat, health, info, ingest, query
from src.api.tasks import IngestJobStore
from src.config import AppConfig
from src.core.rag_service import RAGApplication
from src.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared singletons on startup; clean up on shutdown."""
    config = AppConfig.load()
    configure_logging(config.log_level)

    rag_app = RAGApplication(config)
    job_store = IngestJobStore()

    app.state.rag_app = rag_app
    app.state.job_store = job_store

    yield


def create_app() -> FastAPI:
    """FastAPI application factory for runtime and tests."""
    app = FastAPI(
        title="RAG Demo API",
        description=(
            "REST API for the RAG document analysis system. "
            "Supports document ingestion, semantic query, and streaming chat."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health.router, tags=["health"])
    app.include_router(info.router, tags=["info"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(query.router, tags=["query"])
    app.include_router(chat.router, tags=["chat"])
    return app
