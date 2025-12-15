# RAG Demo

Sistema de Retrieval-Augmented Generation com LangChain, ChromaDB e Ollama.

## Visao geral
- Ingestao de documentos (PDF/TXT/MD), splitting e embeddings com sentence-transformers.
- Vector store persistente com Chroma.
- Chain RAG configuravel (LLM Ollama, top-k, temperatura).
- CLI para ingestao, queries e informacoes do sistema.

Documentacao detalhada em `docs/` (arquitetura, guias, FAQ, changelog) ou veja o [README da pasta docs](docs/README.md).

## Requisitos
- Python 3.12
- Ollama instalado e modelo (ex.: `ollama pull llama3`)
- Ambiente virtual recomendado

## Instalacao rapida
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
copy .env.example .env  # ou ajuste variaveis
```

## Uso rapido
```bash
# Indexar documentos em ./data para ./vectorstore
python main.py ingest

# Iniciar CLI interativa de consulta
python main.py query --interactive

# Consulta unica
python main.py query -q "Qual o assunto principal?"
```

## Verificacao rapida
```bash
# Validar imports basicos
python -c "from src.app import RAGApplication; print('OK')"

# Verificar ambiente e caminhos
python main.py info
```

## Checklist de ambiente
- Ollama instalado e em execucao (`ollama list` deve responder).
- Modelo Ollama baixado (ex.: `ollama pull llama3`).
- Permissao de rede habilitada para baixar embeddings na primeira execucao.
- Pasta `data/` com arquivos ou subpastas; permissao de escrita em `vectorstore/`.
- Ambiente virtual ativo antes de rodar CLI.

## Configuracao (env)
Principais variaveis (veja `.env.example`):
- `DATA_DIR` / `VECTORSTORE_DIR`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_DOCUMENTS`, `TEMPERATURE`
- `LOG_LEVEL`
- `USE_GPU`, `GPU_DEVICE` (opcional, para aceleracao GPU)

## Aceleracao por GPU (Opcional)

A aplicacao usa **CPU por padrao**, mas pode ser acelerada com GPU NVIDIA (CUDA):

### Requisitos
- GPU NVIDIA compativel (GTX 1060+, RTX serie)
- CUDA Toolkit 11.8+ instalado
- Drivers NVIDIA atualizados

### Instalacao
```bash
# Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar disponibilidade
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Configuracao
Adicione ao `.env`:
```bash
USE_GPU=true
GPU_DEVICE=0  # ID da GPU (0, 1, 2...)
```

### Performance
- **Embeddings**: 3-5x mais rapido com GPU
- **LLM (Ollama)**: Usa GPU automaticamente se disponivel
- **Recomendado**: 6GB+ VRAM para modelos grandes

### Monitorar uso da GPU
```bash
# Durante ingestao/query:
nvidia-smi -l 1
```

## CLI (comandos)
- `python main.py ingest` — indexa arquivos (opcoes: `--file-types`, `--chunk-size`, `--chunk-overlap`).
- `python main.py query` — modo interativo (`--interactive`) ou pergunta unica (`-q`), ajustando `--model`, `--embedding-model`, `--top-k`, `--temperature`.
- `python main.py info` — mostra paths, modelos e status do vectorstore.

## Estrutura do projeto
```
src/
  app.py            # Orquestracao (RAGApplication)
  config.py         # Config e loader de env
  logging_config.py # Configuracao de logging
  ingest.py         # Ingestao de documentos
  chain.py          # Builder/factory da chain RAG
  query.py          # Interface de consulta e CLI interativa
tests/
  unit/             # Testes unitarios
  integration/      # Testes de integracao leves
docs/               # Guias, arquitetura, FAQ, changelog
```

## Testes
```bash
pytest tests/unit -q
pytest tests/integration -q
```

## Suporte e links uteis
- Documentacao detalhada: [README da pasta docs](docs/README.md).
- Issues/Discussoes: https://github.com/patrickmcruz/rag-demo
- Email: patrickmcruz@gmail.com
