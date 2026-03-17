# Suite de Testes

Suite de testes do RAG Demo. Cobre toda a aplicacao — modulos de negocio e camada de servico FastAPI — com testes unitarios e de integracao. Nenhum teste requer chamadas reais a Ollama, ChromaDB ou modelos de embeddings.

## Resultado atual

```
67 passed, 0 failed
```

## Como rodar

```bash
# Suite completa
.venv/Scripts/python -m pytest tests/ -v

# Com relatorio de cobertura
.venv/Scripts/python -m pytest tests/ --cov=src --cov-report=term-missing

# Apenas testes rapidos (exclui marcados como slow)
.venv/Scripts/python -m pytest tests/ -m "not slow" -v
```

## Categorias

| Pasta | Testes | Descricao |
|-------|--------|-----------|
| [core/unit/](core/unit/README.md) | 16 | Modulos de negocio: config, app, chain, ingest, query |
| [core/integration/](core/integration/README.md) | 1 | Fiacao ponta-a-ponta (RAGApplication sem LLM/DB real) |
| [api/](api/README.md) | 50 | Camada FastAPI: schemas, job store e todos os endpoints |

## Infraestrutura

### conftest.py (raiz de tests/)

- Adiciona o root do projeto ao `sys.path` para que `src.*` seja importavel em todos os testes.
- Define `pytest_configure` que aponta `basetemp` para `.pytest_tmp/` (local e gitignored), evitando erros de permissao no diretorio temporario do sistema no Windows.
- Fornece a fixture `app_config` compartilhada entre `tests/core/` e `tests/api/`.

### Fixtures compartilhadas

A fixture `app_config` esta definida em `tests/conftest.py` e disponivel para todos os tests. As fixtures especificas da API estao em [tests/api/conftest.py](api/conftest.py). As fixtures gerais (`tmp_path`, `monkeypatch`) sao providas pelo proprio pytest.

### Classes auxiliares (tests/helpers.py)

Dummies reutilizaveis entre `core/` e `api/`: `DummyIngestor`, `DummyIngestionService`, `DummyChain` (com `astream()`), `DummyFactory`, `DummyQuery`.

### Marcadores disponıveis

```bash
# Testes lentos (nenhum ativo por padrao)
pytest tests/ -m slow

# Testes de integracao
pytest tests/ -m integration
```

## Convencoes

- **Sem dependencias externas reais**: todos os testes usam `monkeypatch` ou objetos dummy.
- **Nomenclatura**: `test_<o_que_faz>.py` → `test_<cenario>()`.
- **Falhas esperadas** sao testadas com `pytest.raises` ou verificacao de status HTTP.
- `AppConfig` sempre recebe `use_gpu=False, gpu_device=0` nas fixtures (campos obrigatorios).
