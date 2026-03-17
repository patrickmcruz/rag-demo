# Testes de API (FastAPI)

Testes da camada de servico REST construida com FastAPI. Cobrem schemas Pydantic, controle de estado de jobs de ingestao e todos os endpoints HTTP.

**50 testes | 0 falhas**

## Categorias

| Pasta | Testes | Descricao |
|-------|--------|-----------|
| [unit/](unit/) | 23 | Validacao de schemas e IngestJobStore |
| [integration/](integration/) | 27 | Endpoints HTTP via TestClient |

## Como rodar

```bash
# Todos os testes de API
.venv/Scripts/python -m pytest tests/api/ -v

# Apenas unitarios
.venv/Scripts/python -m pytest tests/api/unit/ -v

# Apenas integracao
.venv/Scripts/python -m pytest tests/api/integration/ -v
```

## Fixtures (conftest.py)

Definidas em `tests/api/conftest.py`. A fixture `app_config` (base para `mock_rag_app`) esta em `tests/conftest.py`.

| Fixture | Escopo | Descricao |
|---------|--------|-----------|
| `app_config` | function | `AppConfig` com `tmp_path` como diretorios (definida em `tests/conftest.py`) |
| `mock_rag_app` | function | `RAGApplication` real, sem chain pre-construida |
| `mock_job_store` | function | `IngestJobStore` vazio |
| `api_client` | function | `TestClient` do FastAPI com `mock_rag_app` e `mock_job_store` injetados em `app.state` |

### Estrategia de injecao

O `TestClient` executa o lifespan da FastAPI ao entrar (`__enter__`), criando os singletons reais em `app.state`. Imediatamente apos, as fixtures sobrescrevem `app.state.rag_app` e `app.state.job_store` com os mocks. Como todos os endpoints acessam os singletons via DI (`get_rag_app`, `get_job_store`), a substituicao e transparente para o codigo das rotas.

```python
with TestClient(fastapi_app) as client:
    fastapi_app.state.rag_app = mock_rag_app    # override pos-lifespan
    fastapi_app.state.job_store = mock_job_store
    yield client
```

## Subcategorias

### [unit/](unit/)

- **`test_models.py`** (14 testes) — validacao dos schemas Pydantic (`QueryRequest`, `IngestRequest`, `ChatRequest`): campos obrigatorios, limites, enums de linguagem.
- **`test_tasks.py`** (9 testes) — maquina de estado de `IngestJob` e thread-safety do `IngestJobStore` (50 threads concorrentes).

### [integration/](integration/)

- **`test_health.py`** (4 testes) — `GET /health` (liveness) e `GET /health/ready` (readiness com e sem vectorstore).
- **`test_info.py`** (5 testes) — `GET /info`: reflexo da config, contagem de arquivos, flags de existencia de diretorios.
- **`test_ingest.py`** (6 testes) — `POST /ingest` (202, conflito 409, validacao de payload) e `GET /ingest/{job_id}` (200, 404).
- **`test_query.py`** (7 testes) — `POST /query`: happy path, 503 sem vectorstore, validacao de entrada (vazio, oversized, linguagem invalida).
- **`test_chat.py`** (5 testes) — `POST /chat/stream`: SSE content-type, evento final `done=true`, eventos intermediarios `done=false`, 503 sem vectorstore, 422 em questao vazia.
