# Suite de Testes

Suite de testes do RAG Demo. Cobre a camada de dominio em `src/core/` e a camada HTTP em `src/api/`, sem depender de chamadas reais a Ollama, ChromaDB ou modelos de embeddings.

## Resultado atual

```text
66 passed
```

## Como rodar

```bash
# Suite completa
.venv/Scripts/python -m pytest tests/ -q

# Com mais detalhes
.venv/Scripts/python -m pytest tests/ -v

# Com cobertura
.venv/Scripts/python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Organizacao

| Pasta | Descricao |
|-------|-----------|
| [core/unit/](core/unit/README.md) | Testes unitarios dos servicos em `src/core/` e da configuracao |
| [core/integration/](core/integration/README.md) | Integracao leve do fluxo de dominio |
| [api/](api/README.md) | Testes de schemas, estado de jobs e endpoints FastAPI |

## Infraestrutura

### [tests/conftest.py](/c:/Users/patrickcruz/Documents/2026/Pessoal/Github/rag-demo/software-engineering/development/tests/conftest.py)

- adiciona a raiz do projeto ao `sys.path`
- fornece uma fixture `tmp_path` controlada pela suite para evitar problemas de permissao no Windows
- fornece a fixture compartilhada `app_config`

### [tests/helpers.py](/c:/Users/patrickcruz/Documents/2026/Pessoal/Github/rag-demo/software-engineering/development/tests/helpers.py)

Contem doubles reutilizaveis entre `core/` e `api/`:
- `DummyIngestor`
- `DummyIngestionService`
- `DummyChain`
- `DummyFactory`
- `DummyQuery`

## Convencoes

- testes de dominio importam de `src/core/*`
- testes de API importam de `src/api/main.py`, `src/api/schemas.py` e `src/api/tasks.py`
- cenarios que antes criavam diretorios reais agora preferem simular existencia de paths ou usar caminhos controlados pela fixture
- marcadores de integracao usam `@pytest.mark.integration`
