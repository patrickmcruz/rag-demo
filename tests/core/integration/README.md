# Testes de Integracao de Negocio

Testes de integracao leve do modulo `src/app.py` (RAGApplication) ponta-a-ponta, sem chamar Ollama, ChromaDB ou modelos de embeddings reais. A chain e substituida por `DummyChain` via `monkeypatch`.

**1 teste | 0 falhas**

## Arquivos

| Arquivo | Testes | Descricao |
|---------|--------|-----------|
| `test_app_light.py` | 1 | RAGApplication.query() com chain mockada; valida fluxo completo |

## Como rodar

```bash
.venv/Scripts/python -m pytest tests/core/integration/ -v -m integration
```

## Marcadores

Todos os testes neste diretorio sao marcados com `@pytest.mark.integration`. Para excluir da suite rapida:

```bash
.venv/Scripts/python -m pytest tests/ -m "not integration" -v
```
