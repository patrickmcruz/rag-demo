"""Query interface for the RAG system."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from a RAG query."""

    answer: str
    sources: List[Document]
    query: str
    response_time: float
    model_name: str
    retrieval_scores: Optional[List[float]] = None

    def format_sources(self) -> str:
        """Format sources for display."""
        if not self.sources:
            return "Nenhuma fonte encontrada."

        formatted = "\n\nFontes:\n"
        for i, doc in enumerate(self.sources, 1):
            source = doc.metadata.get("source", "Desconhecida")
            content_preview = doc.page_content[:150].replace("\n", " ")
            formatted += f"{i}. {source}\n   {content_preview}...\n"
        return formatted

    def __str__(self) -> str:
        return (
            f"Resposta: {self.answer}\n"
            f"{self.format_sources()}\n"
            f"[TIME] Tempo de resposta: {self.response_time:.2f}s\n"
            f"[MODEL] Modelo: {self.model_name}"
        )


class RAGQuery:
    """High-level interface for querying the RAG system."""

    def __init__(self, rag_chain, model_name: str = "llama3"):
        self.rag_chain = rag_chain
        self.model_name = model_name
        self.query_history: List[Dict[str, Any]] = []

    def query(
        self,
        question: str,
        return_sources: bool = True,
        verbose: bool = False,
    ) -> RAGResponse:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if verbose:
            logger.info(f"Query: {question}")

        start_time = time.time()

        try:
            answer = self.rag_chain.invoke(question)
            sources: List[Document] = []
            if return_sources:
                sources = self._get_sources(question)

            response_time = time.time() - start_time
            response = RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                response_time=response_time,
                model_name=self.model_name,
            )

            self.query_history.append(
                {
                    "query": question,
                    "answer": answer,
                    "timestamp": time.time(),
                    "response_time": response_time,
                }
            )

            if verbose:
                logger.info(f"Response time: {response_time:.2f}s")

            return response

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error during query: {exc}")
            raise

    def _get_sources(self, question: str) -> List[Document]:
        try:
            if hasattr(self.rag_chain, "first"):
                retriever_dict = self.rag_chain.first
                if "context" in retriever_dict:
                    retriever = retriever_dict["context"]
                    return retriever.get_relevant_documents(question)  # type: ignore[no-any-return]
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not retrieve sources: {exc}")
            return []

    def batch_query(self, questions: List[str], verbose: bool = False) -> List[RAGResponse]:
        logger.info(f"Processing {len(questions)} questions")
        responses = []
        for i, question in enumerate(questions, 1):
            if verbose:
                logger.info(f"Processing question {i}/{len(questions)}")
            responses.append(self.query(question, verbose=verbose))
        logger.info(f"Completed {len(responses)} queries")
        return responses

    def get_stats(self) -> Dict[str, Any]:
        if not self.query_history:
            return {"total_queries": 0}
        response_times = [q["response_time"] for q in self.query_history]
        return {
            "total_queries": len(self.query_history),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
        }

    def clear_history(self) -> None:
        self.query_history.clear()
        logger.info("Query history cleared")
