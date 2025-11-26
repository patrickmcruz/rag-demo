# Arquitetura do Sistema RAG Demo

Este documento descreve a arquitetura técnica completa do sistema RAG (Retrieval-Augmented Generation).

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Componentes Principais](#componentes-principais)
- [Fluxo de Dados](#fluxo-de-dados)
- [Stack Tecnológica](#stack-tecnológica)
- [Decisões de Design](#decisões-de-design)

---

## 🎯 Visão Geral

O **RAG Demo** é um sistema de recuperação e geração aumentada que permite fazer perguntas sobre documentos usando um LLM local. A arquitetura segue o padrão RAG moderno com três fases principais:

1. **Ingestão**: Processar e indexar documentos
2. **Recuperação**: Buscar contexto relevante
3. **Geração**: Produzir respostas com LLM

```
┌─────────────────────────────────────────────────────────────┐
│                     SISTEMA RAG DEMO                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  INGESTÃO    │───▶│  INDEXAÇÃO   │───▶│ VECTORSTORE  │ │
│  │  (ingest.py) │    │  (ChromaDB)  │    │  (Chroma)    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                                         ▲         │
│         │                                         │         │
│         ▼                                         │         │
│  ┌──────────────┐                                │         │
│  │  DOCUMENTOS  │                                │         │
│  │ (PDF/TXT/MD) │                                │         │
│  └──────────────┘                                │         │
│                                                   │         │
│  ┌──────────────┐    ┌──────────────┐           │         │
│  │   QUERY      │───▶│  RETRIEVAL   │───────────┘         │
│  │  (query.py)  │    │  (chain.py)  │                     │
│  └──────────────┘    └──────────────┘                     │
│         │                     │                             │
│         │                     ▼                             │
│         │            ┌──────────────┐                      │
│         │            │     LLM      │                      │
│         │            │   (Ollama)   │                      │
│         │            └──────────────┘                      │
│         │                     │                             │
│         ▼                     ▼                             │
│  ┌─────────────────────────────────┐                      │
│  │         RESPOSTA FINAL          │                      │
│  │   (answer + sources + metadata) │                      │
│  └─────────────────────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Componentes Principais

### 1. Pipeline de Ingestão (`src/ingest.py`)

**Responsabilidade**: Processar documentos e criar índice vetorial.

#### Classe: `DocumentIngestor`

```python
class DocumentIngestor:
    """Handles document ingestion and indexing for RAG."""
    
    def __init__(self, embedding_model, chunk_size, chunk_overlap):
        # Configuração do modelo de embeddings e parâmetros
        
    def load_documents(self, data_dir, file_types) -> List[Document]:
        # Carrega documentos de múltiplos formatos
        
    def split_documents(self, documents) -> List[Document]:
        # Divide documentos em chunks menores
        
    def create_vectorstore(self, documents, persist_dir) -> Chroma:
        # Cria e persiste vector store
        
    def ingest(self, data_dir, persist_dir, file_types) -> Chroma:
        # Pipeline completa de ingestão
```

#### Fluxo de Ingestão

```
Documentos (data/)
      ↓
[DirectoryLoader] → Carrega arquivos (.pdf, .txt, .md)
      ↓
[RecursiveCharacterTextSplitter] → Divide em chunks
      ↓
[HuggingFaceEmbeddings] → Gera embeddings
      ↓
[ChromaDB] → Indexa e persiste
      ↓
VectorStore (vectorstore/)
```

**Parâmetros de Splitting**:
- `chunk_size`: 500 caracteres (balanceio contexto/granularidade)
- `chunk_overlap`: 50 caracteres (evita perda de contexto)
- `separators`: `["\n\n", "\n", " ", ""]` (hierárquico)

### 2. RAG Chain (`src/chain.py`)

**Responsabilidade**: Configurar e executar a chain de recuperação e geração.

#### Classe: `RAGChainBuilder`

```python
class RAGChainBuilder:
    """Builder for creating configurable RAG chains."""
    
    def __init__(self, vectorstore_path, model_name, embedding_model, 
                 temperature, top_k):
        # Configuração da chain
        
    def build_retriever(self):
        # Configura retriever do vectorstore
        
    def build_llm(self):
        # Inicializa LLM (Ollama)
        
    def build_prompt(self, language) -> ChatPromptTemplate:
        # Cria prompt template otimizado
        
    def build(self) -> Runnable:
        # Monta chain completa
```

#### Arquitetura da Chain

```
Query do Usuário
      ↓
[Embedding] → Vetoriza pergunta
      ↓
[Retriever] → Busca top-k documentos similares
      ↓
[Contexto + Query] → Monta prompt
      ↓
[LLM Ollama] → Gera resposta
      ↓
[Output Parser] → Extrai texto
      ↓
Resposta Final
```

**Componentes LangChain**:
```python
chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)
```

### 3. Interface de Query (`src/query.py`)

**Responsabilidade**: Interface de alto nível para consultas.

#### Classe: `RAGQuery`

```python
class RAGQuery:
    """High-level interface for querying the RAG system."""
    
    def __init__(self, chain, model_name):
        # Inicializa interface
        
    def query(self, question: str) -> RAGResponse:
        # Executa query e retorna resposta estruturada
        
    def get_stats(self) -> Dict:
        # Retorna estatísticas de uso
```

#### Dataclass: `RAGResponse`

```python
@dataclass
class RAGResponse:
    answer: str                      # Resposta gerada
    sources: List[Document]          # Documentos fonte
    query: str                       # Query original
    response_time: float             # Tempo de resposta
    model_name: str                  # Modelo usado
    retrieval_scores: List[float]    # Scores de similaridade
```

### 4. CLI Principal (`main.py`)

**Responsabilidade**: Interface de linha de comando.

#### Comandos

```bash
# Ingestão
python main.py ingest [--file-types] [--chunk-size] [--chunk-overlap]

# Query
python main.py query [-q QUESTION | --interactive] 
                    [--top-k] [--temperature]

# Info
python main.py info
```

#### Argumentos Globais

```python
--data-dir          # Diretório com documentos
--vectorstore-dir   # Diretório do vectorstore
--model            # Modelo Ollama a usar
```

---

## 🔄 Fluxo de Dados

### Fase 1: Ingestão (Offline)

```
┌────────────┐
│ Documentos │
│ (PDF/TXT)  │
└─────┬──────┘
      │
      ▼
┌─────────────────────┐
│ Load Documents      │
│ - PyPDFLoader       │
│ - TextLoader        │
│ - UnstructuredMD    │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Split into Chunks   │
│ - Size: 500 chars   │
│ - Overlap: 50 chars │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Generate Embeddings │
│ - all-MiniLM-L6-v2  │
│ - 384 dimensions    │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Index in ChromaDB   │
│ - HNSW algorithm    │
│ - Cosine similarity │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Persist to Disk     │
│ ./vectorstore/      │
└─────────────────────┘
```

### Fase 2: Query (Online)

```
┌────────────┐
│   Query    │
│  "..."     │
└─────┬──────┘
      │
      ▼
┌─────────────────────┐
│ Embed Query         │
│ - Same model        │
│ - 384 dims          │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Similarity Search   │
│ - Cosine distance   │
│ - Top-k=3 docs      │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Build Prompt        │
│ - Context + Query   │
│ - Template PT/EN    │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Call LLM (Ollama)   │
│ - Llama 3 local     │
│ - Temperature: 0.0  │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Parse Response      │
│ - Extract answer    │
│ - Add metadata      │
└─────┬───────────────┘
      │
      ▼
┌─────────────────────┐
│ Return RAGResponse  │
│ - Answer + Sources  │
└─────────────────────┘
```

---

## 🛠️ Stack Tecnológica

### Core Framework
- **LangChain 1.1.0**: Orquestração da pipeline RAG
  - `langchain-core`: Abstrações base
  - `langchain-chroma`: Integração ChromaDB
  - `langchain-ollama`: Integração Ollama
  - `langchain-huggingface`: Embeddings
  - `langchain-community`: Loaders e utilitários

### Vector Store
- **ChromaDB >= 0.5.0**: Banco vetorial
  - Algoritmo: HNSW (Hierarchical Navigable Small World)
  - Métrica: Similaridade por cosseno
  - Persistência: Disco local
  - Performance: O(log n) para busca

### Embeddings
- **HuggingFace Sentence Transformers**
  - Modelo: `all-MiniLM-L6-v2`
  - Dimensões: 384
  - Tamanho: ~90MB
  - Velocidade: ~50 sentenças/segundo (CPU)
  - Multilíngue: Suporte PT-BR

### LLM
- **Ollama**: Runtime local para LLMs
  - Modelo padrão: Llama 3 (8B)
  - Alternativas: Mistral, Phi, Llama 2
  - Quantização: 4-bit (Q4_0)
  - Interface: REST API local (port 11434)

### Processamento de Documentos
- **pypdf 3.17.4**: Extração de PDFs
- **unstructured 0.11.8**: Parser multi-formato
- **python-magic-bin**: Detecção de tipos (Windows)

### Infraestrutura
- **Python 3.9+**: Linguagem base
- **NumPy 1.26.4**: Operações vetoriais (fixado para compatibilidade)
- **python-dotenv**: Gerenciamento de configuração

---

## 🎨 Decisões de Design

### 1. **Por que Ollama + Llama 3?**

**Vantagens**:
- ✅ **100% Local**: Privacidade total, sem envio de dados
- ✅ **Zero Custo**: Sem custos de API
- ✅ **Offline**: Funciona sem internet
- ✅ **Flexível**: Fácil trocar modelos
- ✅ **Open Source**: Transparência total

**Trade-offs**:
- ⚠️ Requer hardware local (GPU recomendada)
- ⚠️ Mais lento que APIs cloud
- ⚠️ Context window menor que GPT-4

### 2. **Por que ChromaDB?**

**Vantagens**:
- ✅ **Simples**: API Python nativa
- ✅ **Persistente**: Salva em disco automaticamente
- ✅ **Rápido**: HNSW algorithm eficiente
- ✅ **Leve**: Sem servidor separado necessário
- ✅ **Integrado**: Suporte nativo LangChain

**Alternativas consideradas**:
- ~~FAISS~~: Sem persistência nativa
- ~~Pinecone~~: Pago, cloud-only
- ~~Weaviate~~: Complexo para uso local

### 3. **Por que all-MiniLM-L6-v2?**

**Vantagens**:
- ✅ **Pequeno**: ~90MB (vs. 1GB+ de modelos maiores)
- ✅ **Rápido**: Embeddings em tempo real
- ✅ **Multilíngue**: Bom suporte PT-BR
- ✅ **Qualidade**: Performance competitiva
- ✅ **Popular**: Bem testado e documentado

**Benchmark**:
```
Modelo              | Tamanho | Dimensões | Velocidade | Qualidade
--------------------|---------|-----------|------------|----------
all-MiniLM-L6-v2    | 90MB    | 384       | ⭐⭐⭐⭐⭐      | ⭐⭐⭐⭐
all-mpnet-base-v2   | 438MB   | 768       | ⭐⭐⭐        | ⭐⭐⭐⭐⭐
multilingual-e5     | 560MB   | 768       | ⭐⭐⭐        | ⭐⭐⭐⭐⭐
```

### 4. **Chunk Size: 500 caracteres**

**Razão**: Balanceio entre contexto e granularidade

```python
# Muito pequeno (200):
# ❌ Perde contexto
# ✅ Busca precisa

# Muito grande (1000):
# ✅ Mantém contexto
# ❌ Busca imprecisa

# Ideal (500):
# ✅ Contexto suficiente
# ✅ Granularidade boa
# ✅ Performance balanceada
```

### 5. **Top-K = 3 documentos**

**Razão**: Sweet spot entre contexto e ruído

```python
# K = 1:  Pode perder contexto importante
# K = 3:  Contexto suficiente sem ruído ✅
# K = 5:  Mais contexto, mas pode ter irrelevante
# K = 10: Muito ruído, confunde o LLM
```

### 6. **Temperature = 0.0**

**Razão**: Respostas determinísticas e factuais

```python
# Temperature 0.0:
# ✅ Respostas consistentes
# ✅ Mais factual
# ✅ Menos alucinações
# ❌ Menos criativo

# Temperature 0.7+:
# ✅ Mais criativo
# ❌ Menos consistente
# ❌ Mais alucinações
```

---

## 📊 Performance e Escalabilidade

### Métricas Atuais

| Operação | Tempo | Throughput |
|----------|-------|------------|
| Ingestão (1 PDF, 50 páginas) | ~10s | ~5 páginas/s |
| Embedding (por documento) | ~100ms | ~10 docs/s |
| Query (top-3) | ~1-3s | Depende do LLM |
| Indexação (100 chunks) | ~2s | ~50 chunks/s |

### Bottlenecks

1. **LLM (Ollama)**: Principal gargalo (~1-2s por query)
   - **Solução**: GPU, modelos menores (phi), quantização
   
2. **Embeddings**: Segundo gargalo (~100ms por chunk)
   - **Solução**: Cache, batch processing, GPU
   
3. **I/O**: Carregamento de PDFs
   - **Solução**: Processamento paralelo

### Escalabilidade

**Documentos**:
- ✅ Atual: ~100-1000 documentos (testado)
- ✅ Estimado: ~10,000 documentos (sem re-arquitetura)
- ⚠️ >10,000: Considerar Pinecone ou Weaviate

**Queries**:
- ✅ Atual: 1 usuário (CLI local)
- ✅ Possível: ~10 usuários simultâneos (API REST)
- ⚠️ >100: Requer load balancing e cache

---

## 🔐 Segurança e Privacidade

### Princípios

1. **Local-First**: Todos dados permanecem na máquina
2. **Zero Cloud**: Sem envio de dados para serviços externos
3. **Open Source**: Código auditável
4. **Telemetry Off**: ChromaDB telemetry desabilitada

### Considerações

- ✅ **Documentos sensíveis**: Seguros (não saem da máquina)
- ✅ **Queries privadas**: Não logadas externamente
- ⚠️ **Logs locais**: Podem conter informações sensíveis
- ⚠️ **Vectorstore**: Contém chunks dos documentos (criptografar disco)

---

## 🔮 Futuras Melhorias

### Curto Prazo
- [ ] Cache de embeddings (evitar re-embedding)
- [ ] Batch processing para ingestão
- [ ] Métricas de qualidade (ragas)

### Médio Prazo
- [ ] API REST com FastAPI
- [ ] Interface web (Streamlit/Gradio)
- [ ] Suporte para mais formatos (DOCX, HTML)
- [ ] Multi-tenancy (múltiplos vectorstores)

### Longo Prazo
- [ ] Fine-tuning de embeddings
- [ ] Hybrid search (vetorial + keyword)
- [ ] Re-ranking com cross-encoder
- [ ] Streaming de respostas

---

## 📚 Referências

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/)
- [RAG Best Practices (Pinecone)](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

**Versão do Documento**: 1.0.0  
**Última Atualização**: Novembro 2025
