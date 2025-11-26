"""RAG chain configuration module.

This module defines and configures the RAG (Retrieval-Augmented Generation)
chain with support for different LLMs, embeddings, and observability.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGChainBuilder:
    """Builder for creating configurable RAG chains."""
    
    def __init__(
        self,
        vectorstore_path: str,
        model_name: str = "llama3",
        embedding_model: str = "all-MiniLM-L6-v2",
        temperature: float = 0.0,
        top_k: int = 3,
    ):
        """Initialize RAG chain builder.
        
        Args:
            vectorstore_path: Path to persisted vector store
            model_name: Name of the Ollama model to use
            embedding_model: Name of HuggingFace embedding model
            temperature: LLM temperature (0.0 = deterministic)
            top_k: Number of documents to retrieve
        """
        self.vectorstore_path = vectorstore_path
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_k = top_k
        
        # Validate vectorstore exists
        if not Path(vectorstore_path).exists():
            raise ValueError(f"Vector store not found at: {vectorstore_path}")
    
    def build_retriever(self):
        """Build and configure the retriever."""
        logger.info(f"Loading vector store from {self.vectorstore_path}")
        
        embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vectorstore = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=embedding
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        logger.info(f"Retriever configured with top_k={self.top_k}")
        return retriever
    
    def build_llm(self):
        """Build and configure the LLM."""
        logger.info(f"Initializing LLM: {self.model_name}")
        
        llm = Ollama(
            model=self.model_name,
            temperature=self.temperature,
        )
        
        return llm
    
    def build_prompt(self, language: str = "pt") -> ChatPromptTemplate:
        """Build the prompt template.
        
        Args:
            language: Language for the prompt ('pt' or 'en')
            
        Returns:
            Configured ChatPromptTemplate
        """
        if language == "pt":
            template = """Você é um assistente útil que responde perguntas com base em documentos fornecidos.

Use APENAS as informações do contexto abaixo para responder à pergunta.
Se você não souber a resposta com base no contexto, diga "Não tenho informações suficientes para responder essa pergunta."

Contexto:
{context}

Pergunta: {question}

Resposta detalhada:"""
        else:
            template = """You are a helpful assistant that answers questions based on provided documents.

Use ONLY the information from the context below to answer the question.
If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Detailed answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def build(self, language: str = "pt"):
        """Build the complete RAG chain.
        
        Args:
            language: Language for prompts ('pt' or 'en')
            
        Returns:
            Configured RAG chain ready for use
        """
        logger.info("Building RAG chain...")
        
        # Build components
        retriever = self.build_retriever()
        llm = self.build_llm()
        prompt = self.build_prompt(language)
        
        # Format context from retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build the chain with RunnableParallel for better tracking
        rag_chain = (
            RunnableParallel(
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain built successfully")
        return rag_chain


def create_rag_chain(
    vectorstore_path: str,
    model_name: str = "llama3",
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.0,
    top_k: int = 3,
    language: str = "pt",
):
    """Convenience function to create a RAG chain.
    
    Args:
        vectorstore_path: Path to persisted vector store
        model_name: Name of the Ollama model to use
        embedding_model: Name of HuggingFace embedding model
        temperature: LLM temperature (0.0 = deterministic)
        top_k: Number of documents to retrieve
        language: Language for prompts ('pt' or 'en')
        
    Returns:
        Configured RAG chain
        
    Example:
        >>> chain = create_rag_chain("./vectorstore")
        >>> response = chain.invoke("What is the main topic?")
    """
    builder = RAGChainBuilder(
        vectorstore_path=vectorstore_path,
        model_name=model_name,
        embedding_model=embedding_model,
        temperature=temperature,
        top_k=top_k,
    )
    return builder.build(language)


def get_chain_info(chain) -> Dict[str, Any]:
    """Get information about a configured chain.
    
    Args:
        chain: Configured RAG chain
        
    Returns:
        Dictionary with chain configuration details
    """
    info = {
        "chain_type": "RAG",
        "components": [],
    }
    
    # Try to extract component information
    try:
        if hasattr(chain, 'steps'):
            info["components"] = [str(step) for step in chain.steps]
    except:
        pass
    
    return info


if __name__ == "__main__":
    # Example usage
    print("Este módulo deve ser importado para criar chains RAG.")
    print("\nExemplo de uso:")
    print("""
    from src.chain import create_rag_chain
    
    # Criar chain
    chain = create_rag_chain(
        vectorstore_path="./vectorstore",
        model_name="llama3",
        top_k=3
    )
    
    # Fazer pergunta
    response = chain.invoke("Qual é o assunto principal?")
    print(response)
    """)