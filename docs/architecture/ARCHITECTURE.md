# Arquitetura do Sistema RAG Demo

Este documento descreve o system design da versao atual do RAG Demo, sua decomposicao em componentes, tecnologias adotadas e os principais trade-offs arquiteturais.

## Sumario
- Visao Geral
- Objetivos Arquiteturais
- Escopo da Solucao
- Contexto do Sistema
- Containers e Componentes
- Fluxos Principais
- Modelo de Dados e Estado
- Tecnologias e Papel de Cada Uma
- Escalabilidade e Limites Atuais
- Seguranca e Operacao
- ADRs Relacionados

## Visao Geral
O RAG Demo e uma aplicacao API-first para ingestao de documentos e consultas semanticas sobre uma base vetorial local. O sistema recebe documentos, cria embeddings, persiste esses vetores em um vector store e expõe endpoints HTTP para consulta sincronica, status operacional e streaming de resposta.

O design privilegia simplicidade operacional, baixa dependencia externa e clareza de separacao entre:
- camada HTTP
- regras de negocio
- pipeline de ingestao
- orquestracao da chain RAG

## Objetivos Arquiteturais
- oferecer uma API simples para ingestao e consulta de documentos
- manter a regra de negocio desacoplada do framework web
- permitir execucao local com LLM via Ollama
- persistir embeddings localmente sem infraestrutura distribuida obrigatoria
- suportar evolucao futura para observabilidade, filas externas e multiworker

## Escopo da Solucao
Inclui:
- ingestao de arquivos `PDF`, `TXT` e `MD`
- indexacao vetorial com embeddings locais
- consulta RAG sincronica
- streaming de resposta via SSE
- health checks e informacoes operacionais

Nao inclui nesta versao:
- autenticacao e autorizacao
- processamento distribuido de jobs
- armazenamento compartilhado de estado de ingestao
- multitenancy
- banco relacional para metadados de aplicacao

## Contexto do Sistema

```text
Cliente HTTP / Frontend / curl
            |
            v
      FastAPI Service
            |
            +--> RAGApplication
            |      |
            |      +--> IngestionService
            |      |      |
            |      |      +--> DocumentIngestor
            |      |             +--> Loaders
            |      |             +--> Text Splitter
            |      |             +--> Embeddings
            |      |             +--> ChromaDB
            |      |
            |      +--> RAGChainFactory
            |             +--> Retriever
            |             +--> Prompt
            |             +--> Ollama
            |
            +--> IngestJobStore (memoria)
```

## Containers e Componentes

### 1. Entry Point HTTP
- Arquivo: `api_server.py`
- Responsabilidade: iniciar a aplicacao FastAPI e carregar configuracao de ambiente antes da criacao do app.

### 2. API Layer
- Pasta: `src/api/`
- Responsabilidade: exposicao HTTP, schemas, dependencias, roteamento e tratamento de erros.

Componentes:
- `main.py`: application factory e ciclo de vida do app
- `routers/*.py`: endpoints por dominio
- `schemas.py`: contratos de request e response
- `deps.py`: resolucao de dependencias a partir de `app.state`
- `tasks.py`: controle em memoria de jobs de ingestao
- `exceptions.py`: handlers de excecao da API

### 3. Core Layer
- Pasta: `src/core/`
- Responsabilidade: regras de negocio e orquestracao do fluxo RAG, sem acoplamento com FastAPI.

Componentes:
- `rag_service.py`: fachada principal da aplicacao
- `ingest_service.py`: pipeline de ingestao e indexacao
- `chain_factory.py`: construcao da chain RAG
- `query_service.py`: execucao de consultas e formatacao de resposta

### 4. Configuracao e Logging
- `src/config.py`: leitura centralizada de configuracao via ambiente
- `src/logging_config.py`: padronizacao de logging

## Fluxos Principais

### Fluxo 1: Startup da API
1. `api_server.py` carrega `.env`
2. `src/api/main.py` cria o `FastAPI`
3. o lifespan inicializa `AppConfig`
4. o lifespan cria `RAGApplication`
5. o lifespan cria `IngestJobStore`
6. ambos sao registrados em `app.state`

### Fluxo 2: Ingestao de Documentos
1. cliente envia `POST /ingest`
2. a API valida o payload com `schemas.py`
3. o endpoint cria um job no `IngestJobStore`
4. a task em background aplica overrides no `RAGApplication`
5. `IngestionService` carrega documentos e divide em chunks
6. embeddings sao gerados
7. os vetores sao persistidos no ChromaDB
8. o status do job e atualizado

### Fluxo 3: Consulta Sincronica
1. cliente envia `POST /query`
2. a API valida que o vectorstore existe
3. `RAGApplication` reaproveita ou recria a chain
4. o retriever busca contexto relevante
5. o prompt e montado
6. o Ollama gera a resposta
7. `RAGQuery` retorna resposta estruturada com fontes e metricas

### Fluxo 4: Streaming de Resposta
1. cliente envia `POST /chat/stream`
2. a API garante readiness do `RAGApplication`
3. a chain e obtida via `get_chain()`
4. o endpoint usa `StreamingResponse`
5. tokens sao enviados em formato SSE

## Modelo de Dados e Estado

### Estado Persistente
- documentos de entrada em `DATA_DIR`
- base vetorial persistida em `VECTORSTORE_DIR`

### Estado em Memoria
- `RAGApplication` em `app.state`
- `IngestJobStore` em `app.state`
- chain RAG cacheada no `RAGApplication`

### Implicacoes
- o estado de jobs de ingestao nao e compartilhado entre processos
- a versao atual deve operar com `workers=1`
- para multiworker, o job store deve migrar para backend compartilhado

## Tecnologias e Papel de Cada Uma

### FastAPI
Escolhida para a camada HTTP por:
- tipagem forte com Pydantic
- baixa friccao para APIs REST
- suporte nativo a async
- documentacao automatica com Swagger e ReDoc

### Uvicorn
Escolhido como servidor ASGI por:
- integracao direta com FastAPI
- simplicidade operacional
- bom suporte a desenvolvimento com reload

### LangChain
Escolhida para orquestracao da pipeline RAG por:
- composicao de prompts, runnables e retrievers
- integracao pronta com Ollama, Chroma e embeddings
- facilidade de evolucao futura para novos retrievers e chains

### ChromaDB
Escolhido como vector store por:
- persistencia local simples
- baixa complexidade de setup
- integracao direta com LangChain

### sentence-transformers / HuggingFaceEmbeddings
Escolhidos por:
- execucao local
- boa cobertura de modelos de embeddings
- facilidade para uso com CPU ou GPU

### Ollama
Escolhido para inferencia local por:
- operacao simples para LLM local
- baixo atrito para ambiente de desenvolvimento
- capacidade de manter dados fora de um provedor remoto

## Escalabilidade e Limites Atuais

### Pontos fortes
- arquitetura simples e facil de manter
- baixo custo operacional
- facil evolucao da camada de API e da camada core separadamente

### Limites conhecidos
- `IngestJobStore` em memoria impede escalabilidade horizontal
- uso de filesystem local para vectorstore limita distribuicao
- ausencia de fila externa limita robustez do processamento assincrono
- ausencia de autenticacao restringe uso em ambientes expostos

## Seguranca e Operacao

### Seguranca
- nao ha autenticacao nativa nesta versao
- CORS e configurado por ambiente
- o uso recomendado e ambiente interno ou protegido por gateway

### Operacao
- `GET /health` para liveness
- `GET /health/ready` para readiness
- logs centralizados por `logging_config.py`
- Swagger em `/docs`
- ReDoc em `/redoc`

## ADRs Relacionados
- [ADR-001-adr-template.md](adr/ADR-001-adr-template.md)
- [ADR-002-api-first-fastapi.md](adr/ADR-002-api-first-fastapi.md)
- [ADR-003-layered-api-core-structure.md](adr/ADR-003-layered-api-core-structure.md)
- [ADR-004-ollama-local-llm.md](adr/ADR-004-ollama-local-llm.md)
- [ADR-005-chromadb-vector-store.md](adr/ADR-005-chromadb-vector-store.md)
- [ADR-006-langchain-rag-orchestration.md](adr/ADR-006-langchain-rag-orchestration.md)
