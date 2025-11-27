# Arquitetura do Sistema RAG Demo

Este documento descreve a arquitetura tecnica do sistema RAG Demo.

## Indice
- Visao Geral
- Componentes Principais
- Fluxo de Dados
- Stack Tecnologica
- Decisoes de Design

---

## Visao Geral
O sistema segue o padrao RAG: ingerir documentos, recuperar contexto relevante e gerar respostas com um LLM local (Ollama).

---

## Componentes Principais

### 0. Orquestracao e Configuracao
- `src/app.py`: `RAGApplication` centraliza ingestao, construcao da chain e consultas (com reset e overrides via CLI).
- `src/config.py`: `AppConfig` / `ConfigManager` carregam variaveis de ambiente e resolvem paths.
- `src/logging_config.py`: `LoggingConfigurator` define logging global.
- `src/query.py`: `InteractiveQueryCLI` separa o loop interativo da logica de negocio.
- `tests/`: reorganizado em `tests/unit/` e `tests/integration/` com `conftest.py` para importar `src`.

### 1. Pipeline de Ingestao (`src/ingest.py`)
Responsavel por carregar documentos, dividir em chunks, gerar embeddings e persistir no vector store.
- `DocumentIngestor`: carrega (PDF, TXT, MD), faz splitting (`chunk_size`, `chunk_overlap`) e cria a vector store.
- `IngestionService`: orquestra o pipeline de ingestao.

### 2. RAG Chain (`src/chain.py`)
Configura a chain de recuperacao + geracao.
- `RAGChainBuilder`: monta retriever (Chroma + HuggingFaceEmbeddings), LLM (OllamaLLM), prompt e pipeline.
- `RAGChainFactory`: fabrica para produzir chains configuradas.

### 3. Interface de Consulta (`src/query.py`)
- `RAGQuery`: consulta a chain e retorna `RAGResponse` (resposta, fontes, tempos).
- `InteractiveQueryCLI`: loop interativo no terminal.

### 4. CLI (`main.py`)
- Define comandos `ingest`, `query` e `info`, aplica overrides no `RAGApplication` e aciona servicos.

---

## Fluxo de Dados (alto nivel)
1. Ingestao: arquivos em `data/` -> carregamento -> splitting -> embeddings -> persistencia no Chroma (`vectorstore/`).
2. Query: pergunta do usuario -> retriever (top-k) -> montagem de prompt com contexto -> LLM (Ollama) -> resposta + fontes.

### Sequencia simplificada (CLI)
1. CLI (`main.py`) recebe comando (`ingest`, `query`, `info`).
2. `RAGApplication` aplica overrides e aciona serviços.
3. Ingestao: `IngestionService` -> `DocumentIngestor` -> Chroma.
4. Query: `RAGApplication` -> `RAGChainFactory` -> chain -> `RAGQuery` -> `InteractiveQueryCLI` (se interativo).
5. Resposta estruturada (texto + fontes + metricas).

---

## Stack Tecnologica
- Python 3.12
- LangChain (chain, prompts, runnables)
- ChromaDB (vector store)
- sentence-transformers (embeddings)
- OllamaLLM (LLM local)

---

## Decisoes de Design
- Orquestracao central via `RAGApplication` para aplicar overrides e cachear chain.
- Configuracao e logging centralizados para evitar duplicacao de `basicConfig`.
- Reorganizacao de testes em unitario e integracao leve para facilitar CI.
