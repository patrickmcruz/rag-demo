# Testes de Negocio (Core)

Testes dos modulos de dominio da aplicacao RAG: configuracao, orquestracao, pipeline de ingestao, chain LangChain e querying. Nenhum teste requer chamadas reais a Ollama, ChromaDB ou modelos de embeddings.

**17 testes | 0 falhas**

## Categorias

| Pasta | Testes | Descricao |
|-------|--------|-----------|
| [unit/](unit/README.md) | 16 | Testes unitarios dos modulos `src/` |
| [integration/](integration/README.md) | 1 | Fiacao ponta-a-ponta sem LLM/DB real |

## Como rodar

```bash
# Todos os testes core
.venv/Scripts/python -m pytest tests/core/ -v

# Apenas unitarios
.venv/Scripts/python -m pytest tests/core/unit/ -v

# Apenas integracao
.venv/Scripts/python -m pytest tests/core/integration/ -v -m integration
```

## Fixtures e helpers

- **`app_config`** (de `tests/conftest.py`) — `AppConfig` pre-configurado com `tmp_path` como diretorios.
- **`dummy_chain`** (de `tests/core/conftest.py`) — instancia de `DummyChain` para testes que precisam de uma chain.
- **`tests/helpers.py`** — classes dummy compartilhadas: `DummyIngestor`, `DummyIngestionService`, `DummyChain`, `DummyFactory`, `DummyQuery`.
