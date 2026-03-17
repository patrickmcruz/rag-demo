"""
FastAPI server entry point.

Run with:
    python api_server.py
    uvicorn api_server:app --reload --port 8000
    uvicorn api_server:app --host 0.0.0.0 --port 8080

Environment variables:
    DATA_DIR, VECTORSTORE_DIR, OLLAMA_MODEL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DOCUMENTS, TEMPERATURE,
    LOG_LEVEL, USE_GPU, GPU_DEVICE

API-specific env vars:
    API_HOST        Server bind host (default: 0.0.0.0)
    API_PORT        Server bind port (default: 8000)
    API_RELOAD      Enable hot reload (default: false, dev only)
    CORS_ORIGINS    Comma-separated allowed origins (default: *)

Workers:
    Run with workers=1 (default). The IngestJobStore is in-process memory.
    For multi-worker deployments, replace IngestJobStore with a Redis-backed store.
"""

import os

from dotenv import load_dotenv

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import uvicorn  # noqa: E402

from src.api.main import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        workers=1,
    )
