# Changelog

Todas as mudancas notaveis neste projeto serao documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/) e o projeto adere ao [Versionamento Semantico](https://semver.org/lang/pt-BR/).

## [Nao Lancado]

### Em Desenvolvimento
- API REST para consultas remotas
- Interface web com Streamlit
- Suporte para mais formatos de documento (DOCX, HTML)
- Cache de embeddings para melhor performance
- Orquestracao central via `RAGApplication` (`src/app.py`)
- Configuracao/logging unificados (`config.py`, `logging_config.py`)
- Reestruturacao dos testes em `tests/unit` e `tests/integration` com `conftest.py`
- Guia de testes (docs/guides/testing.md) e link entre README raiz e docs
- Checklist de ambiente e verificacao rapida no README

---

## [1.0.0] - 2025-11-26

### Lancamento Inicial

Primeira versao estavel do sistema RAG Demo com as funcionalidades principais.

### Adicionado

#### Infraestrutura e Configuracao
- Estrutura modular do projeto (`src/`, `tests/`, `docs/`)
- Ambiente virtual Python com dependencias gerenciadas
- Arquivo `.env` para configuracoes
- Sistema de logging estruturado
- Suporte multiplataforma (Windows, Linux, macOS)

#### Pipeline de Ingestao (`src/ingest.py`)
- Carregamento de documentos (PDF, TXT, Markdown)
- Splitting de texto com `RecursiveCharacterTextSplitter`
- Embeddings com HuggingFace (all-MiniLM-L6-v2)
- Indexacao persistente com ChromaDB
- Validacoes e tratamento de erros
- Logging detalhado do processo

#### RAG Chain (`src/chain.py`)
- Configuracao flexivel da chain RAG
- Suporte para modelos Ollama (Llama 3, Mistral, Phi, etc.)
- Retriever configuravel (top-k, similarity search)
- Prompts em portugues e ingles
- Validacao de vectorstore antes de queries

#### Interface de Query (`src/query.py`)
- CLI interativa e modo de consulta unica
- Respostas estruturadas com metadados e fontes
- Historico e estatisticas de consultas
- Metricas de performance (tempo de resposta)

#### CLI Principal (`main.py`)
- Comando `ingest` para indexacao
- Comando `query` (interativo ou unica)
- Comando `info` para informacoes do sistema
- Argumentos configuraveis (model, temperature, top-k, etc.)

#### Documentacao
- README principal
- `docs/FAQ.md`
- `docs/ARCHITECTURE.md`
- Guias em `docs/guides/` (inicio rapido, modelos, embeddings, troubleshooting)

#### Testes
- Testes unitarios para componentes principais
- Testes de integracao da pipeline
- Cobertura de codigo com pytest-cov
