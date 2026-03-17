# Testes de Core

Testes da camada de dominio em `src/core/` e da configuracao da aplicacao. Nenhum teste depende de chamadas reais a LLM, vector store ou embeddings.

## Como rodar

```bash
# Todos os testes de core
.venv/Scripts/python -m pytest tests/core/ -q

# Apenas unitarios
.venv/Scripts/python -m pytest tests/core/unit/ -q

# Apenas integracao
.venv/Scripts/python -m pytest tests/core/integration/ -q -m integration
```

## Cobertura

| Pasta | Descricao |
|-------|-----------|
| [unit/](unit/README.md) | `rag_service`, `chain_factory`, `ingest_service`, `query_service` e `config` |
| [integration/](integration/README.md) | fluxo leve ponta-a-ponta do `RAGApplication` |

## Fixtures e helpers

- `app_config`: fixture compartilhada de [tests/conftest.py](/c:/Users/patrickcruz/Documents/2026/Pessoal/Github/rag-demo/software-engineering/development/tests/conftest.py)
- `tests/helpers.py`: doubles reutilizaveis usados pelos testes unitarios e de integracao
