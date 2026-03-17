"""Streaming chat endpoint using Server-Sent Events (SSE)."""

import json
import logging
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import get_ready_rag_app
from src.api.models import ChatRequest
from src.app import RAGApplication

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat")


async def _token_generator(chain, question: str) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted strings.

    SSE format per event:
        data: {"token": "...", "done": false}\\n\\n

    Final event:
        data: {"token": "", "done": true}\\n\\n

    Frontend usage (JavaScript fetch + ReadableStream):
        const res = await fetch('/chat/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: '...'})
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const lines = decoder.decode(value).split('\\n\\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    process(data.token, data.done);
                }
            }
        }
    """
    try:
        async for chunk in chain.astream(question):
            payload = json.dumps({"token": chunk, "done": False})
            yield f"data: {payload}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
    except Exception as exc:
        logger.error(f"Stream error: {exc}")
        payload = json.dumps({"error": str(exc), "done": True})
        yield f"data: {payload}\n\n"


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    rag_app: Annotated[RAGApplication, Depends(get_ready_rag_app)],
) -> StreamingResponse:
    """
    Stream an answer token-by-token using Server-Sent Events.

    The LangChain chain (OllamaLLM | StrOutputParser) natively supports
    .astream(), so no chain modifications are required.
    """
    chain = rag_app.get_chain()
    return StreamingResponse(
        _token_generator(chain, request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
