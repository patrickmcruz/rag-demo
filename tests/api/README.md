# Testes de API

Testes da camada FastAPI em `src/api/`. Cobrem schemas, controle de jobs de ingestao e endpoints HTTP usando `TestClient`.

## Como rodar

```bash
# Todos os testes de API
.venv/Scripts/python -m pytest tests/api/ -q

# Apenas unitarios
.venv/Scripts/python -m pytest tests/api/unit/ -q

# Apenas integracao
.venv/Scripts/python -m pytest tests/api/integration/ -q
```

## Fixtures

As fixtures de API estao em [tests/api/conftest.py](/c:/Users/patrickcruz/Documents/2026/Pessoal/Github/rag-demo/software-engineering/development/tests/api/conftest.py).

| Fixture | Descricao |
|---------|-----------|
| `app_config` | configuracao compartilhada definida em `tests/conftest.py` |
| `mock_rag_app` | instancia real de `RAGApplication` baseada em `src/core/rag_service.py` |
| `mock_job_store` | `IngestJobStore` vazio |
| `api_client` | `TestClient` com `rag_app` e `job_store` injetados em `app.state` |

## Cobertura

### `unit/`

- `test_models.py`: validacao dos schemas em `src/api/schemas.py`
- `test_tasks.py`: comportamento de `IngestJobStore` e `IngestJob`

### `integration/`

- `test_health.py`: `GET /health` e `GET /health/ready`
- `test_info.py`: `GET /info`
- `test_ingest.py`: `POST /ingest` e `GET /ingest/{job_id}`
- `test_query.py`: `POST /query`
- `test_chat.py`: `POST /chat/stream`

## Estrategia

O app e criado por `create_app()` em `src/api/main.py`. Depois do lifespan iniciar, os singletons de `app.state` sao sobrescritos pelos mocks da suite, mantendo o comportamento das rotas sob teste sem usar dependencias externas reais.
