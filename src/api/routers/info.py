"""System info endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.deps import get_rag_app
from src.api.models import InfoResponse
from src.app import RAGApplication

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def system_info(
    rag_app: Annotated[RAGApplication, Depends(get_rag_app)],
) -> InfoResponse:
    """Return current configuration and filesystem statistics."""
    cfg = rag_app.config
    data_dir = cfg.data_dir

    file_counts: dict[str, int] = {}
    if data_dir.exists():
        for ext in ["txt", "pdf", "md"]:
            file_counts[ext] = len(list(data_dir.glob(f"**/*.{ext}")))
    else:
        file_counts = {"txt": 0, "pdf": 0, "md": 0}

    return InfoResponse(
        model=cfg.model,
        embedding_model=cfg.embedding_model,
        data_dir=str(cfg.data_dir),
        vectorstore_dir=str(cfg.vectorstore_dir),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        top_k_documents=cfg.top_k_documents,
        temperature=cfg.temperature,
        use_gpu=cfg.use_gpu,
        gpu_device=cfg.gpu_device,
        vectorstore_exists=cfg.vectorstore_dir.exists(),
        data_dir_exists=data_dir.exists(),
        data_file_counts=file_counts,
    )
