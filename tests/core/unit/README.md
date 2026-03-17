# Testes Unitarios de Negocio

Testes unitarios dos modulos `src/`: cada modulo e testado de forma isolada usando objetos dummy e `monkeypatch`. Nenhum acesso a disco, rede ou modelos de ML.

**16 testes | 0 falhas**

## Arquivos

| Arquivo | Testes | Modulo testado |
|---------|--------|----------------|
| `test_config.py` | 4 | `src/config.py` — `AppConfig`, defaults, env overrides, cache |
| `test_app.py` | 4 | `src/app.py` — `RAGApplication`: overrides, ingest, chain cache, query |
| `test_chain_factory.py` | 1 | `src/chain.py` — `RAGChainFactory` com builder dummy |
| `test_ingest_service.py` | 1 | `src/ingest.py` — `IngestionService` delega ao ingestor |
| `test_query.py` | 6 | `src/query.py` — `RAGQuery`: happy path, batch, stats, empty raises |

## Como rodar

```bash
.venv/Scripts/python -m pytest tests/core/unit/ -v
```

## Convencoes

- `AppConfig` sempre criado via fixture `app_config` (de `tests/conftest.py`).
- Falhas esperadas testadas com `pytest.raises`, nunca com `try/except`.
- Dummies importados de `tests/helpers.py` para evitar duplicacao.
