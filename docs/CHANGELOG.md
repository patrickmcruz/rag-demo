# Changelog

Todas as mudanças notáveis neste projeto serão documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/) e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não Lançado]

### Alterado
- O projeto passa a operar somente em modo API.
- O entrypoint principal é `api_server.py`.
- A documentação principal foi reorganizada entre `README.md` e `docs/README.md`.

### Removido
- `main.py`
- fluxo de execução por CLI
- utilitários interativos de terminal ligados a query

## [0.2.0] - 2026-03-17

### Refatoração para FastAPI
- projeto disponibilizado com Fast API em modo de serviço desacoplado do HTTP.
- estrutura de pastas e organização refeitas.
- suite de testes completas para cada tipo de teste automatizado.
- documentação pronta com swagger.
- documentação completa da arquitetura com `adr`
- base de testes

## [0.1.0] - 2025-11-26

### Lançamento Inicial
- estrutura modular do projeto
- pipeline de ingestão com ChromaDB
- chain RAG com LangChain e Ollama
- camada de consulta estruturada
- documentação inicial
- base de testes

