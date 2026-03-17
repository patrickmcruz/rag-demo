# Documentacao do RAG Demo

Guia principal do projeto refatorado com FastAPI como unica interface de execucao.

## Sumario
- [Visao geral](#visao-geral)
- [Tecnologias](#tecnologias)
- [Arquitetura](#arquitetura)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como clonar o repositorio](#como-clonar-o-repositorio)
- [Pre-requisitos](#pre-requisitos)
- [Configuracao do ambiente](#configuracao-do-ambiente)
- [Como rodar a aplicacao](#como-rodar-a-aplicacao)
- [Documentacao da API](#documentacao-da-api)
- [Endpoints principais](#endpoints-principais)
- [Exemplos de uso](#exemplos-de-uso)
- [Testes](#testes)
- [Variaveis de ambiente](#variaveis-de-ambiente)
- [Observacoes de implantacao](#observacoes-de-implantacao)
- [Documentos complementares](#documentos-complementares)

## Visao geral
O RAG Demo e uma aplicacao de Retrieval-Augmented Generation para indexacao de documentos e consulta semantica sobre uma base local. A versao atual expoe a funcionalidade principal exclusivamente via FastAPI, mantendo a camada de dominio desacoplada da interface HTTP.

Capacidades principais:
- ingestao de documentos `PDF`, `TXT` e `MD`
- geracao de embeddings com modelos locais
- persistencia vetorial com ChromaDB
- consultas RAG com LangChain
- execucao local de LLM com Ollama
- streaming de resposta com Server-Sent Events

## Tecnologias
- `Python 3.12`
- `FastAPI`
- `Uvicorn`
- `LangChain`
- `ChromaDB`
- `sentence-transformers`
- `Ollama`
- `pytest`, `pytest-asyncio` e `pytest-cov`

## Arquitetura
Os principais pontos da arquitetura atual sao:
- `api_server.py` expoe a aplicacao para execucao com Uvicorn
- `src/api/main.py` cria a aplicacao FastAPI, registra middleware e rotas
- `src/api/routers/` organiza os endpoints por dominio
- `src/api/deps.py` fornece dependencias compartilhadas
- `src/api/tasks.py` mantem o controle de jobs de ingestao em memoria
- `src/core/rag_service.py` concentra a orquestracao de ingestao, chain e consulta
- `src/core/ingest_service.py` implementa o pipeline de indexacao
- `src/core/chain_factory.py` monta a cadeia RAG
- `src/core/query_service.py` concentra a logica de consulta e retorno estruturado
Maiores detalhes sobre arquitetura estão disponíveis abaixo

Documentacao completa de arquitetura:
- [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)
- [architecture/adr/ADR-001-adr-template.md](architecture/adr/ADR-001-adr-template.md)
- [architecture/adr/ADR-002-api-first-fastapi.md](architecture/adr/ADR-002-api-first-fastapi.md)
- [architecture/adr/ADR-003-layered-api-core-structure.md](architecture/adr/ADR-003-layered-api-core-structure.md)
- [architecture/adr/ADR-004-ollama-local-llm.md](architecture/adr/ADR-004-ollama-local-llm.md)
- [architecture/adr/ADR-005-chromadb-vector-store.md](architecture/adr/ADR-005-chromadb-vector-store.md)
- [architecture/adr/ADR-006-langchain-rag-orchestration.md](architecture/adr/ADR-006-langchain-rag-orchestration.md)

## Estrutura do projeto
```text
api_server.py
requirements.txt
src/
  api/
    main.py
    deps.py
    exceptions.py
    schemas.py
    tasks.py
    routers/
      chat.py
      health.py
      info.py
      ingest.py
      query.py
  core/
    rag_service.py
    chain_factory.py
    ingest_service.py
    query_service.py
  config.py
  logging_config.py
tests/
  api/
  core/
docs/
data/
```

## Como clonar o repositorio
```bash
git clone https://github.com/patrickmcruz/rag-demo.git
cd rag-demo/software-engineering/development
```

## Pre-requisitos
- Python `3.12`
- Ollama instalado localmente
- ao menos um modelo disponivel no Ollama, por exemplo `llama3`
- ambiente virtual recomendado

## Configuracao do ambiente
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Verificacao rapida:
```bash
python -c "from src.api.main import create_app; print(create_app().title)"
```

## Como rodar a aplicacao
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

Alternativa:
```bash
python api_server.py
```

## Documentacao da API
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints principais
- `GET /health`
- `GET /health/ready`
- `GET /info`
- `POST /ingest`
- `GET /ingest/{job_id}`
- `POST /query`
- `POST /chat/stream`

## Exemplos de uso

Usando bash/shell com `curl`:
```bash
curl -X POST "http://localhost:8000/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{\"data_dir\":\"./data\",\"vectorstore_dir\":\"./vectorstore\"}"
```

```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Qual o conteudo principal do documento?\",\"return_sources\":true}"
```

PowerShell com `Invoke-RestMethod`:

```powershell
# Ingestao
$ingestBody = @{
  data_dir = "./data"
  vectorstore_dir = "./vectorstore"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ingest" `
  -ContentType "application/json" `
  -Body $ingestBody
```

```powershell
# Query
$queryBody = @{
  question = "Qual o conteudo principal do documento?"
  return_sources = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/query" `
  -ContentType "application/json" `
  -Body $queryBody
```

## Testes
```bash
pytest -q
pytest tests/api -q
pytest tests/core -q
```

## Variaveis de ambiente
- `DATA_DIR`
- `VECTORSTORE_DIR`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K_DOCUMENTS`
- `TEMPERATURE`
- `LOG_LEVEL`
- `USE_GPU`
- `GPU_DEVICE`
- `API_HOST`
- `API_PORT`
- `API_RELOAD`
- `CORS_ORIGINS`

## Observacoes de implantacao
- O armazenamento de jobs de ingestao e em memoria de processo.
- Para multiplos workers, o ideal e substituir `IngestJobStore` por um backend compartilhado.
- O readiness depende da existencia do vectorstore.
- A aplicacao depende de Ollama local disponivel para consultas reais.

## Documentos complementares
- [README da raiz](../README.md)
- [USAGE.md](USAGE.md)
- [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)
- [CHANGELOG.md](CHANGELOG.md)
- [guides/testing.md](guides/testing.md)
