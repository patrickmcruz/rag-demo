"""Query interface for RAG system.

This module provides a high-level interface for querying the RAG system
with support for metadata, source tracking, and performance metrics.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from RAG query."""

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
        """String representation of response."""
        return (
            f"Resposta: {self.answer}\n"
            f"{self.format_sources()}\n"
            f"[TIME] Tempo de resposta: {self.response_time:.2f}s\n"
            f"[MODEL] Modelo: {self.model_name}"
        )


class RAGQuery:
    """High-level interface for querying RAG system."""

    def __init__(self, rag_chain, model_name: str = "llama3"):
        """Initialize RAG query interface.

        Args:
            rag_chain: Configured RAG chain from chain.py
            model_name: Name of the LLM model being used
        """
        self.rag_chain = rag_chain
        self.model_name = model_name
        self.query_history: List[Dict[str, Any]] = []

    def query(
        self,
        question: str,
        return_sources: bool = True,
        verbose: bool = False,
    ) -> RAGResponse:
        """Query the RAG system.

        Args:
            question: User's question
            return_sources: Whether to return source documents
            verbose: Enable verbose logging

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if verbose:
            logger.info(f"Query: {question}")

        start_time = time.time()

        try:
            # Invoke the RAG chain
            answer = self.rag_chain.invoke(question)

            # Get source documents (if retriever is accessible)
            sources: List[Document] = []
            if return_sources:
                sources = self._get_sources(question)

            response_time = time.time() - start_time

            # Create response object
            response = RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                response_time=response_time,
                model_name=self.model_name,
            )

            # Store in history
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

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error during query: {e}")
            raise

    def _get_sources(self, question: str) -> List[Document]:
        """Retrieve source documents for a question.

        Args:
            question: User's question

        Returns:
            List of source documents
        """
        try:
            if hasattr(self.rag_chain, "first"):
                retriever_dict = self.rag_chain.first
                if "context" in retriever_dict:
                    retriever = retriever_dict["context"]
                    return retriever.get_relevant_documents(question)  # type: ignore[no-any-return]
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not retrieve sources: {e}")
            return []

    def batch_query(
        self,
        questions: List[str],
        verbose: bool = False,
    ) -> List[RAGResponse]:
        """Query multiple questions in batch.

        Args:
            questions: List of questions
            verbose: Enable verbose logging

        Returns:
            List of RAGResponse objects
        """
        logger.info(f"Processing {len(questions)} questions")
        responses = []

        for i, question in enumerate(questions, 1):
            if verbose:
                logger.info(f"Processing question {i}/{len(questions)}")
            response = self.query(question, verbose=verbose)
            responses.append(response)

        logger.info(f"Completed {len(responses)} queries")
        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get query statistics.

        Returns:
            Dictionary with query statistics
        """
        if not self.query_history:
            return {"total_queries": 0}

        response_times = [q["response_time"] for q in self.query_history]

        return {
            "total_queries": len(self.query_history),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
        }

    def clear_history(self):
        """Clear query history."""
        self.query_history.clear()
        logger.info("Query history cleared")


def interactive_query_loop(rag_chain, model_name: str = "llama3"):
    """Interactive CLI for querying the RAG system.

    Args:
        rag_chain: Configured RAG chain
        model_name: Name of the LLM model
    """
    query_interface = RAGQuery(rag_chain, model_name)

    print("\n" + "=" * 60)
    print("[RAG] Query Interface")
    print("=" * 60)
    print("Digite sua pergunta (ou 'sair' para encerrar)")
    print("Comandos especiais:")
    print("  - 'stats': Mostrar estatísticas")
    print("  - 'clear': Limpar histórico")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("\n[QUERY] Pergunta: ").strip()

            if not question:
                continue

            if question.lower() in ["sair", "exit", "quit"]:
                print("\nEncerrando...")
                stats = query_interface.get_stats()
                if stats["total_queries"] > 0:
                    print(f"\nTotal de consultas: {stats['total_queries']}")
                    print(f" Tempo médio: {stats['avg_response_time']:.2f}s")
                break

            if question.lower() == "stats":
                stats = query_interface.get_stats()
                print("\nEstatísticas:")
                for key, value in stats.items():
                    print(
                        f"  {key}: {value if isinstance(value, int) else f'{value:.2f}'}"
                    )
                continue

            if question.lower() == "clear":
                query_interface.clear_history()
                print("Histórico limpo")
                continue

            # Query the RAG system
            response = query_interface.query(question, verbose=True)

            # Display response
            print("\n" + "=" * 60)
            print(f"[ANSWER] Resposta:\n{response.answer}")
            print(response.format_sources())
            print(f"[TIME] Tempo: {response.response_time:.2f}s")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n[EXIT] Encerrando...")
            break
        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro: {e}")
            print(f"\n[ERROR] Erro: {e}")


if __name__ == "__main__":
    # Example usage
    print("Este módulo deve ser importado e usado com uma chain RAG configurada.")
    print("\nExemplo de uso:")
    print(
        """
    from src.chain import create_rag_chain
    from src.query import interactive_query_loop

    chain = create_rag_chain("./vectorstore")
    interactive_query_loop(chain)
    """
    )
