"""RAG chain configuration module."""

import logging
from pathlib import Path
from typing import Any, Dict

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


def get_device(use_gpu: bool = False, gpu_device: int = 0) -> str:
    """Detect and return the appropriate device for embeddings."""
    if not use_gpu:
        logger.info("GPU disabled by configuration, using CPU")
        return "cpu"

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, falling back to CPU")
        return "cpu"

    if torch.cuda.is_available():
        device = f"cuda:{gpu_device}"
        gpu_name = torch.cuda.get_device_name(gpu_device)
        logger.info(f"GPU available: {gpu_name}, using {device}")
        return device

    logger.warning("CUDA not available, falling back to CPU")
    return "cpu"


class RAGChainBuilder:
    """Builder for creating configurable RAG chains."""

    def __init__(
        self,
        vectorstore_path: str,
        model_name: str = "llama3",
        embedding_model: str = "all-MiniLM-L6-v2",
        temperature: float = 0.0,
        top_k: int = 3,
        use_gpu: bool = False,
        gpu_device: int = 0,
    ):
        self.vectorstore_path = vectorstore_path
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_k = top_k
        self.device = get_device(use_gpu, gpu_device)

        if not Path(vectorstore_path).exists():
            raise ValueError(f"Vector store not found at: {vectorstore_path}")

    def build_retriever(self):
        logger.info(f"Loading vector store from {self.vectorstore_path}")
        logger.info(f"Using device: {self.device}")

        embedding = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            persist_directory=self.vectorstore_path, embedding_function=embedding
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )

        logger.info(f"Retriever configured with top_k={self.top_k}")
        return retriever

    def build_llm(self):
        logger.info(f"Initializing LLM: {self.model_name}")
        return OllamaLLM(model=self.model_name, temperature=self.temperature)

    def build_prompt(self, language: str = "pt") -> ChatPromptTemplate:
        if language == "pt":
            template = """Você é um assistente especializado em análise de documentos legais e editais.

📋 TAREFA: Responder completamente à pergunta com TODAS as informações disponíveis.

⚠️ INSTRUÇÕES CRÍTICAS:
1. LEIA TODO o contexto fornecido
2. LISTE TODOS os itens relevantes (não apenas alguns)
3. Se a pergunta pede lista -> SEMPRE use formato numerado
4. Se há múltiplos itens similares -> LISTE TODOS SEM EXCEÇÃO
5. Se a resposta estiver incompleta no contexto, indique "Ver documento para lista completa"
6. Cite PÁGINA ou SEÇÃO quando possível

📄 CONTEXTO DO DOCUMENTO:
{context}

❓ PERGUNTA DO USUÁRIO:
{question}

✅ RESPOSTA COMPLETA E DETALHADA:"""
        else:
            template = """You are a legal document and tender analysis specialist.

📋 TASK: Answer the question completely with ALL available information.

⚠️ CRITICAL INSTRUCTIONS:
1. READ ALL the provided context
2. LIST ALL relevant items (not just some)
3. For listing requests -> ALWAYS use numbered format
4. If there are multiple similar items -> LIST ALL WITHOUT EXCEPTION
5. If the answer seems incomplete, add "See document for complete list"
6. Cite PAGE or SECTION when possible

📄 DOCUMENT CONTEXT:
{context}

❓ USER QUESTION:
{question}

✅ COMPLETE AND DETAILED ANSWER:"""

        return ChatPromptTemplate.from_template(template)

    def build(self, language: str = "pt"):
        logger.info("Building RAG chain...")

        retriever = self.build_retriever()
        llm = self.build_llm()
        prompt = self.build_prompt(language)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnableParallel(
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("[OK] RAG chain built successfully")
        return rag_chain


class RAGChainFactory:
    """Factory that produces configured RAG chains."""

    def __init__(
        self,
        vectorstore_path: str,
        model_name: str = "llama3",
        embedding_model: str = "all-MiniLM-L6-v2",
        temperature: float = 0.0,
        top_k: int = 3,
        language: str = "pt",
        use_gpu: bool = False,
        gpu_device: int = 0,
    ):
        self.builder = RAGChainBuilder(
            vectorstore_path=vectorstore_path,
            model_name=model_name,
            embedding_model=embedding_model,
            temperature=temperature,
            top_k=top_k,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        self.language = language

    def create(self):
        return self.builder.build(self.language)


def create_rag_chain(
    vectorstore_path: str,
    model_name: str = "llama3",
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.0,
    top_k: int = 3,
    language: str = "pt",
    use_gpu: bool = False,
    gpu_device: int = 0,
):
    """Convenience function to create a RAG chain using the factory."""
    factory = RAGChainFactory(
        vectorstore_path=vectorstore_path,
        model_name=model_name,
        embedding_model=embedding_model,
        temperature=temperature,
        top_k=top_k,
        language=language,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    return factory.create()


def get_chain_info(chain) -> Dict[str, Any]:
    """Get information about a configured chain."""
    info = {"chain_type": "RAG", "components": []}
    try:
        if hasattr(chain, "steps"):
            info["components"] = [str(step) for step in chain.steps]
    except Exception:  # noqa: BLE001
        pass
    return info
