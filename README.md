# RAG Demo - Sistema RAG Profissional com LangChain

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Sistema de Retrieval-Augmented Generation (RAG) profissional usando LangChain, Chroma e Ollama.

</div>

## 📋 Índice

- [Sobre](#-sobre)
- [Arquitetura](#-arquitetura)
- [Funcionalidades](#-funcionalidades)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Testes](#-testes)
- [Configuração](#-configuração)
- [Próximos Passos](#-próximos-passos)

## 🎯 Sobre

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) completo e profissional, seguindo as melhores práticas de engenharia de software:

- ✅ **Modular e testável**: Código organizado com separação clara de responsabilidades
- ✅ **Logging e validações**: Tratamento de erros robusto e logs informativos
- ✅ **Suporte multi-formato**: PDF, TXT, Markdown
- ✅ **Configurável**: Variáveis de ambiente para todas as configurações
- ✅ **Documentado**: Docstrings completas e type hints
- ✅ **Preparado para produção**: Estrutura escalável e manutenível

## 🏗️ Arquitetura

```
[Usuário]
   ↓ (pergunta)
[Query Interface (query.py)]
   ↓
1. Embedding da pergunta
   ↓
2. Retrieval no Chroma (top-k documentos)
   ↓
3. Montagem do prompt com contexto
   ↓
4. Chamada ao LLM (Ollama)
   ↓
5. Resposta + metadados (fontes, tempo)
   ↓
[Resposta Estruturada]
```

### Componentes principais:

- **ingest.py**: Carrega, processa e indexa documentos
- **chain.py**: Define e configura a chain RAG
- **query.py**: Interface de alto nível para consultas

## ✨ Funcionalidades

### Ingestion Pipeline
- ✅ Carregamento de múltiplos formatos (TXT, PDF, MD)
- ✅ Splitting inteligente de documentos
- ✅ Embeddings com HuggingFace (sentence-transformers)
- ✅ Indexação persistente com Chroma
- ✅ Logging detalhado de todo o processo

### RAG Chain
- ✅ Configuração flexível (temperatura, top-k, etc.)
- ✅ Suporte para múltiplos modelos Ollama
- ✅ Prompts otimizados (PT/EN)
- ✅ Validações e tratamento de erros

### Query Interface
- ✅ CLI interativo
- ✅ Respostas estruturadas com metadados
- ✅ Rastreamento de fontes
- ✅ Métricas de performance
- ✅ Histórico de consultas

## 🔧 Pré-requisitos

### Sistema
- Python 3.9+
- 4GB+ RAM (para embeddings)
- ~2GB de espaço em disco

### Software necessário
1. **Ollama** (para LLM local)
   ```bash
   # Instalar Ollama: https://ollama.ai
   # Baixar modelo:
   ollama pull llama3
   ```

2. **Python e pip**
   ```bash
   python --version  # deve ser 3.9+
   ```

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/rag-demo.git
cd rag-demo
```

### 2. Crie um ambiente virtual
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente
```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite .env conforme necessário
```

## 🚀 Uso

### 1. Preparar documentos
Coloque seus documentos na pasta `data/`:
```bash
# Adicione arquivos .txt, .pdf ou .md
cp seus_documentos.pdf data/
```

### 2. Indexar documentos
```python
python -c "
from src.ingest import ingest_documents

# Indexar todos os documentos
ingest_documents(
    data_dir='./data',
    persist_dir='./vectorstore',
    file_types=['txt', 'pdf', 'md']
)
"
```

### 3. Consultar o sistema

#### Modo interativo (CLI)
```python
from src.chain import create_rag_chain
from src.query import interactive_query_loop

# Criar chain
chain = create_rag_chain(vectorstore_path='./vectorstore')

# Iniciar CLI interativo
interactive_query_loop(chain)
```

#### Modo programático
```python
from src.chain import create_rag_chain
from src.query import RAGQuery

# Criar chain
chain = create_rag_chain(
    vectorstore_path='./vectorstore',
    model_name='llama3',
    top_k=3,
    temperature=0.0
)

# Criar interface de query
query = RAGQuery(chain, model_name='llama3')

# Fazer pergunta
response = query.query("Qual é o assunto principal dos documentos?")
print(response)

# Ver estatísticas
print(query.get_stats())
```

## 📁 Estrutura do Projeto

```
rag-demo/
├── .env                    # Configurações (não versionado)
├── .env.example            # Exemplo de configurações
├── .gitignore              # Arquivos ignorados pelo git
├── requirements.txt        # Dependências Python
├── README.md              # Este arquivo
│
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Pipeline de ingestão
│   ├── chain.py           # Configuração da chain RAG
│   └── query.py           # Interface de consulta
│
├── data/                  # Documentos fonte
│   └── (seus arquivos)
│
├── vectorstore/           # Base vetorial persistida
│   └── (gerado automaticamente)
│
└── tests/
    ├── __init__.py
    └── test_rag.py        # Testes unitários e integração
```

## 🧪 Testes

### Executar todos os testes
```bash
pytest tests/ -v
```

### Executar testes específicos
```bash
# Apenas testes rápidos
pytest tests/ -v -m "not slow"

# Testes de integração
pytest tests/ -v -m integration

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

### Executar linting
```bash
# Formatar código
black src/ tests/

# Verificar estilo
flake8 src/ tests/

# Type checking
mypy src/
```

## ⚙️ Configuração

### Variáveis de ambiente (.env)

```bash
# Modelo Ollama
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434

# Modelo de Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Caminhos
DATA_DIR=./data
VECTORSTORE_DIR=./vectorstore

# Configuração RAG
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_DOCUMENTS=3
TEMPERATURE=0.0

# Logging
LOG_LEVEL=INFO
```

### Modelos Ollama suportados
- `llama3` (recomendado)
- `phi3`
- `mistral`
- `codellama`

Para instalar um modelo:
```bash
ollama pull <model-name>
```

## 🔮 Próximos Passos

### Curto prazo
- [ ] Adicionar suporte a mais formatos (DOCX, HTML)
- [ ] Implementar cache de embeddings
- [ ] Adicionar CLI com argparse
- [ ] Melhorar prompts para casos específicos

### Médio prazo
- [ ] Integrar LangSmith para observabilidade
- [ ] Adicionar avaliação com RAGAS
- [ ] Implementar API REST com FastAPI
- [ ] Adicionar suporte a Vertex AI (produção)

### Longo prazo
- [ ] Interface web (Streamlit/Gradio)
- [ ] Suporte a conversas (chat com memória)
- [ ] Multi-tenancy
- [ ] Deploy com Docker

## 📚 Recursos Adicionais

### Documentação
- [LangChain Docs](https://python.langchain.com/)
- [Ollama](https://ollama.ai/)
- [Chroma](https://docs.trychroma.com/)
- [LangSmith](https://docs.smith.langchain.com/)

### Artigos relacionados
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Faça fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## ✍️ Autor

Desenvolvido com ❤️ para demonstrar boas práticas em sistemas RAG.

---

**Nota**: Este é um projeto educacional/demonstrativo. Para uso em produção, considere adicionar autenticação, rate limiting, monitoramento e outras funcionalidades enterprise.
