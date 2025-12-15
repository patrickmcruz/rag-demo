# Documentacao RAG Demo

Bem-vindo a documentacao do projeto **RAG Demo** - sistema de Retrieval-Augmented Generation com LangChain, ChromaDB e Ollama.

## Documentos principais
- **[FAQ](FAQ.md)**: duvidas frequentes sobre modelos, embeddings, tokenizacao e troubleshooting.
- **[Arquitetura do Sistema](ARCHITECTURE.md)**: visao tecnica, componentes e fluxo de dados.
- **[Changelog](CHANGELOG.md)**: historico de mudancas e versoes.

## Guias praticos
- **[Inicio Rapido](guides/quickstart.md)**: configure e execute o projeto em minutos.
- **[Resolucao de Problemas](guides/troubleshooting.md)**: solucoes para erros comuns.
- **[Guia de Modelos LLM](guides/models.md)**: escolha e configuracao de modelos Ollama.
- **[Guia de Embeddings](guides/embeddings.md)**: como otimizar embeddings.
- **[Aceleracao por GPU](guides/gpu-acceleration.md)**: configure CUDA para performance maxima.
- **[Testes](guides/testing.md)**: como executar e criar testes.

## Por onde comecar
- Novo no projeto? Leia o [README principal](../README.md), siga o [Inicio Rapido](guides/quickstart.md) e consulte o [FAQ](FAQ.md).
- Quer entender a fundo? Estude a [Arquitetura](ARCHITECTURE.md), explore os guias e veja o [Changelog](CHANGELOG.md).
- Com problemas? Consulte o [Troubleshooting](guides/troubleshooting.md), o [FAQ](FAQ.md) ou abra uma issue.

## Estrutura da documentacao
```
docs/
  README.md                # Este indice
  FAQ.md                   # Perguntas frequentes
  ARCHITECTURE.md          # Arquitetura do sistema
  CHANGELOG.md             # Historico de mudancas
  guides/
    quickstart.md          # Inicio rapido
    embeddings.md          # Guia de embeddings
    models.md              # Guia de modelos LLM
    troubleshooting.md     # Resolucao de problemas
    gpu-acceleration.md    # Aceleracao por GPU
    testing.md             # Guia de testes
```

## Recursos do projeto
- Codigo: `src/ingest.py`, `src/chain.py`, `src/query.py`, `src/app.py`, `src/config.py`, `src/logging_config.py`.
- Exemplos: `examples/`
- Scripts: `scripts/`
- Testes: `tests/unit/`, `tests/integration/`, `tests/conftest.py`

## Variaveis de ambiente (principais)
| Variavel           | Default            | Descricao                                    |
| ------------------ | ------------------ | -------------------------------------------- |
| `DATA_DIR`         | `./data`           | Pasta com documentos                         |
| `VECTORSTORE_DIR`  | `./vectorstore`    | Pasta de persistencia do Chroma              |
| `OLLAMA_MODEL`     | `llama3`           | Modelo Ollama                                |
| `EMBEDDING_MODEL`  | `all-MiniLM-L6-v2` | Modelo de embeddings HuggingFace             |
| `CHUNK_SIZE`       | `500`              | Tamanho de chunk                             |
| `CHUNK_OVERLAP`    | `50`               | Sobreposicao entre chunks                    |
| `TOP_K_DOCUMENTS`  | `3`                | Documentos retornados pelo retriever         |
| `TEMPERATURE`      | `0.0`              | Temperatura do LLM                           |
| `LOG_LEVEL`        | `INFO`             | Nivel de log                                 |
| `USE_GPU`          | `false`            | Habilitar aceleracao GPU                     |
| `GPU_DEVICE`       | `0`                | ID do dispositivo GPU                        |

## Contribuindo
1. Use linguagem clara e exemplos praticos.
2. Siga o padrao Markdown dos arquivos existentes.
3. Use links relativos.
4. Inclua exemplos de codigo quando relevante.
5. Atualize o Changelog.

## Suporte
- Issues: [GitHub Issues](https://github.com/patrickmcruz/rag-demo/issues)
- Discussoes: [GitHub Discussions](https://github.com/patrickmcruz/rag-demo/discussions)
- Email: patrickmcruz@gmail.com

---

Ultima atualizacao: Novembro 2025  
Versao da documentacao: 1.0.0
