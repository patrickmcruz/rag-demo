# Guia de Testes

## Comandos principais
- Testes unitarios: `pytest tests/unit -q`
- Testes de integracao: `pytest tests/integration -q`
- Testes lentos ou marcados: `pytest -m "slow"` ou `pytest -m "integration"`
- Cobertura: `pytest --cov=src --cov-report=term-missing`

## Notas
- `tests/conftest.py` adiciona o caminho raiz ao `sys.path` para importar `src`.
- Testes de integracao leve usam dummies para evitar dependencia de LLM/Chroma externos.
- Em Windows, feche handles antes de apagar `vectorstore` gerados em testes.
