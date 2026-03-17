# RAG Demo

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orquestracao-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-6E59F7)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLM%20Local-111111)](https://ollama.com/)

Sistema de Retrieval-Augmented Generation para ingestao de documentos e consultas semanticas com API em FastAPI, embeddings locais e execucao de modelos via Ollama.

## Resumo
O projeto opera exclusivamente em modo API. A aplicacao expoe endpoints para ingestao, consulta, health check, informacoes do sistema e streaming de resposta, mantendo a logica central reutilizavel em uma camada de dominio separada.

## Destaques
- API REST com FastAPI
- Chat com streaming via SSE
- Ingestao assincrona com controle de status
- ChromaDB como vector store persistente
- Ollama para execucao local de LLM
- Camada `core` separada da camada `api`
- Base de testes separada entre `tests/api` e `tests/core`

## Apresentacao do projeto
Na versao atual, todo o fluxo principal e API-first:
- `api_server.py` inicia o servico HTTP
- `src/api/main.py` monta a aplicacao FastAPI
- `src/core/rag_service.py` concentra a orquestracao principal do RAG

## Estrutura de documentacao
A documentacao detalhada do projeto esta em [docs/README.md](docs/README.md).

La voce encontra:
- sumario tecnico
- stack e tecnologias detalhadas
- arquitetura e system design
- registros de decisoes arquiteturais (ADRs)
- como clonar o repositorio
- como configurar ambiente
- como rodar a API
- endpoints principais e Swagger
- testes
- organizacao do projeto
- links para documentos complementares

## Acesso rapido
- Documentacao completa: [docs/README.md](docs/README.md)
- Arquitetura: [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)
- ADRs: [docs/architecture/adr/](docs/architecture/adr)
- Changelog: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Testes automatizados: [tests/README.md](tests/README.md)
