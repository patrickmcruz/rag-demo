# Arquitetura do Sistema RAG Demo

Este documento descreve o system design da versao atual do RAG Demo, sua decomposicao em componentes, tecnologias adotadas e os principais trade-offs arquiteturais.

## Sumario
- Visao Geral
- Objetivos Arquiteturais
- Escopo da Solucao
- Contexto do Sistema
- Containers e Componentes
- Fluxos Principais
- Diagramas de Sequencia
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

## Diagramas de Sequencia

Diagrama principal dos fluxos de interacao do usuario com os endpoints:
- [api-user-flows-sequence.mmd](diagrams/api-user-flows-sequence.mmd)

O diagrama cobre:
- `GET /health`
- `GET /info`
- `GET /health/ready`
- `POST /ingest`
- `GET /ingest/{job_id}`
- `POST /query`
- `POST /chat/stream`

## Fluxo Descritivo Fim a Fim

### 1. Cliente e entrada HTTP
O fluxo sempre comeca no usuario ou frontend, que pode ser um browser, uma SPA, um script PowerShell ou `curl`. Esse ator consome a API por HTTP e representa a camada de apresentacao da solucao.

O primeiro ponto de contato com o sistema e a API FastAPI. O FastAPI foi escolhido para expor os endpoints porque oferece validacao forte de payloads com Pydantic, boa ergonomia para REST e suporte nativo a operacoes assincronas. Na pratica, ele funciona como a porta de entrada do sistema.

### 2. Descoberta do estado do sistema
Antes de consultar ou ingerir documentos, o cliente normalmente chama `GET /health`, `GET /info` e `GET /health/ready`.

Nesse momento:
- `GET /health` verifica se o processo HTTP esta vivo
- `GET /info` retorna configuracao operacional relevante
- `GET /health/ready` verifica se a base vetorial ja existe e se o sistema pode responder consultas

A classe central aqui e `RAGApplication`, em `src/core/rag_service.py`. Ela atua como fachada da aplicacao: centraliza configuracao, acesso ao vector store, criacao da chain e execucao do fluxo RAG. Em vez de cada endpoint falar diretamente com varios componentes, a API conversa com essa fachada.

### 3. Ingestao de documentos
Se o sistema ainda nao estiver pronto, o cliente chama `POST /ingest`. O endpoint recebe o payload, valida os campos com `schemas.py` e cria um job no `IngestJobStore`.

O `IngestJobStore`, em `src/api/tasks.py`, e um armazenamento em memoria para controlar o ciclo de vida da ingestao. Ele registra estados como `queued`, `running`, `done` e `failed`. Sua funcao e dar visibilidade operacional ao frontend enquanto a ingestao acontece em background.

Depois disso, a API aciona o `RAGApplication`, que aplica overrides de execucao e delega o trabalho ao `IngestionService`, em `src/core/ingest_service.py`.

O `IngestionService` e a peca responsavel por transformar documentos brutos em uma base pesquisavel. Ele:
- carrega arquivos do disco
- normaliza e corrige texto quando necessario
- divide o conteudo em chunks
- gera embeddings
- persiste os vetores no banco vetorial

### 4. Papel das tecnologias na ingestao
Durante a ingestao, algumas tecnologias entram em cena com papeis bem definidos.

Os loaders leem arquivos `PDF`, `TXT` e `MD`. O text splitter quebra documentos grandes em partes menores para melhorar recuperacao e contexto. Sem chunking, a busca vetorial e a qualidade do contexto pioram.

Os embeddings sao gerados via `sentence-transformers` usando `HuggingFaceEmbeddings`. O papel dessa tecnologia e converter texto em vetores numericos que preservam similaridade semantica. E isso que permite buscar por significado, e nao apenas por palavras exatas.

O `ChromaDB` funciona como vector store local. Ele persiste os embeddings e os metadados dos chunks para que consultas futuras possam recuperar os trechos mais relevantes. Foi escolhido porque oferece persistencia local simples e baixa complexidade operacional.

### 5. Polling do job de ingestao
Enquanto a ingestao roda, o frontend consulta `GET /ingest/{job_id}`. Esse endpoint nao executa a ingestao em si; ele apenas le o estado do `IngestJobStore`.

Essa separacao e importante: o cliente nao precisa manter uma conexao bloqueada esperando a indexacao terminar. Em vez disso, ele faz polling e atualiza a interface de acordo com o progresso observado.

### 6. Consulta RAG sincronica
Quando o vector store ja existe, o cliente pode chamar `POST /query`. O endpoint valida o payload, verifica readiness e entrega a pergunta ao `RAGApplication`.

O `RAGApplication` reutiliza uma chain existente ou pede ao `RAGChainFactory`, em `src/core/chain_factory.py`, para montar uma nova chain. O `RAGChainFactory` tem a funcao de compor os elementos da pipeline RAG:
- retriever
- prompt
- modelo generativo
- parser de saida

O retriever consulta o `ChromaDB` para buscar os chunks semanticamente mais relevantes para a pergunta. Esses trechos sao inseridos no prompt como contexto.

Em seguida, o `Ollama` entra como mecanismo de inferencia local. O papel dele e executar o modelo de linguagem localmente, gerando a resposta final sem depender de um provedor externo. Isso reduz dependencia externa e favorece privacidade dos dados.

Por fim, o `query_service.py` organiza a resposta em um formato consistente, incluindo:
- resposta final
- fontes utilizadas
- tempo de resposta
- nome do modelo

### 7. Streaming de resposta
No endpoint `POST /chat/stream`, o fluxo e parecido com o `/query`, mas a forma de entrega muda.

Em vez de esperar a resposta completa, a API usa `StreamingResponse` para enviar eventos SSE conforme o modelo gera novos tokens. Nesse fluxo:
- o frontend inicia a conexao
- a chain produz tokens incrementalmente
- a API encapsula esses tokens em eventos `data: ...`
- o cliente renderiza a resposta em tempo real

Aqui, o FastAPI cumpre dois papeis: manter a conexao aberta e serializar os eventos SSE. O Ollama continua sendo o gerador dos tokens. A chain continua sendo o componente que une recuperacao de contexto e geracao da resposta.

### 8. Estado e limites arquiteturais
Ao longo do fluxo inteiro, dois tipos de estado coexistem.

O estado persistente fica no filesystem:
- documentos em `DATA_DIR`
- vetores em `VECTORSTORE_DIR`

O estado de execucao fica em memoria:
- `RAGApplication`
- chain cacheada
- `IngestJobStore`

Isso simplifica muito o design atual, mas tambem explica um limite importante: como o `IngestJobStore` fica em memoria local, a aplicacao foi desenhada para operar com um unico worker. Em uma evolucao futura, jobs e estado operacional precisariam migrar para um backend compartilhado, como Redis ou banco de dados.

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
