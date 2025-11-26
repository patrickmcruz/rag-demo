# RAG Demo - Sistema RAG Profissional com LangChain

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)
![License](https://img.shields.io/badge/License-GNU%20GPL-blue)

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
- [FAQ](#-faq)

## 🎯 Sobre

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) completo e profissional, seguindo as melhores práticas de engenharia de software:

- **Modular e testável**: Código organizado com separação clara de responsabilidades
- **Logging e validações**: Tratamento de erros robusto e logs informativos
- **Suporte multi-formato**: PDF, TXT, Markdown
- **Configurável**: Variáveis de ambiente para todas as configurações
- **Documentado**: Docstrings completas e type hints
- **Preparado para produção**: Estrutura escalável e manutenível

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
- Carregamento de múltiplos formatos (TXT, PDF, MD)
- Splitting inteligente de documentos
- Embeddings com HuggingFace (sentence-transformers)
- Indexação persistente com Chroma
- Logging detalhado de todo o processo

### RAG Chain
- Configuração flexível (temperatura, top-k, etc.)
- Suporte para múltiplos modelos Ollama
- Prompts otimizados (PT/EN)
- Validações e tratamento de erros

### Query Interface
- CLI interativo
- Respostas estruturadas com metadados
- Rastreamento de fontes
- Métricas de performance
- Histórico de consultas

## 🔧 Pré-requisitos

### Sistema
- **Python 3.9+** (testado com Python 3.12)
- **4GB+ RAM** (para embeddings e modelos)
- **~2GB de espaço em disco** (para modelos e índices)
- **Windows 10/11, Linux ou macOS**

### Software Necessário

#### 1. Python e pip
```bash
python --version  # deve ser 3.9 ou superior
pip --version
```

#### 2. Ollama (LLM Local)
**Instalar Ollama:**
- Windows/Mac: Baixe de [https://ollama.ai](https://ollama.ai)
- Linux: `curl -fsSL https://ollama.ai/install.sh | sh`

**Baixar um modelo:**
```bash
# Verificar se Ollama está rodando
ollama list

# Baixar modelo recomendado
ollama pull llama3

# Ou outros modelos disponíveis:
# ollama pull llama2
# ollama pull mistral
# ollama pull phi
```

**Verificar instalação:**
```bash
ollama run llama3 "Hello"
```

#### 3. Microsoft Visual C++ Build Tools (Somente Windows)
**Necessário para compilar algumas dependências Python**

- **Opção 1 (Recomendada):** Baixe e instale: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Durante instalação, selecione "Desktop development with C++"
  
- **Opção 2:** Ative o Developer Mode no Windows
  - Configurações → Atualização e Segurança → Para desenvolvedores → Modo de Desenvolvedor

> **Nota:** Se não instalar, você pode ter erros ao instalar pacotes como `chroma-hnswlib`

### Dependências Python Críticas

O projeto usa as seguintes versões específicas para compatibilidade:

- **NumPy:** `1.26.4` (não use NumPy 2.0+ - incompatível com sentence-transformers)
- **LangChain:** Pacotes atualizados (`langchain-chroma`, `langchain-ollama`, `langchain-huggingface`)
- **ChromaDB:** `>=0.5.0` (corrige problemas de telemetria)
- **sentence-transformers:** Para embeddings locais

> **Importante:** As dependências serão instaladas automaticamente pelo `requirements.txt` com as versões corretas.

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/patrickmcruz/rag-demo.git
cd rag-demo
```

### 2. Crie e ative um ambiente virtual
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# Linux/Mac:
source .venv/bin/activate
```

> **Importante:** Sempre ative o ambiente virtual antes de executar comandos Python!

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

> **Nota:** A instalação inclui:
> - LangChain e componentes atualizados
> - ChromaDB para vector store
> - Sentence Transformers para embeddings
> - NumPy 1.26.4 (compatível)
> - Todas as dependências necessárias

### 4. Configure as variáveis de ambiente
```bash
# Copie o arquivo de exemplo
# Windows:
copy .env.example .env

# Linux/Mac:
cp .env.example .env

# Edite .env conforme necessário (opcional)
```

**Configurações principais no `.env`:**
```env
OLLAMA_MODEL=llama3              # Modelo Ollama a usar
VECTORSTORE_DIR=./vectorstore    # Onde salvar índice
DATA_DIR=./data                  # Pasta com documentos
CHUNK_SIZE=500                   # Tamanho dos chunks
TOP_K_DOCUMENTS=3                # Documentos a recuperar
TEMPERATURE=0.0                  # Temperatura do LLM (0.0 = determinístico)
ANONYMIZED_TELEMETRY=False       # Desabilitar telemetria ChromaDB
```

### 5. Verifique a instalação
```bash
# Verificar se Ollama está rodando
ollama list

# Testar imports Python
python -c "from src.ingest import ingest_documents; print('OK')"
```

## 🚀 Uso

### 1. Preparar documentos
Coloque seus documentos (PDF, TXT, MD) na pasta `data/`:
```bash
# Exemplo: copiar seus PDFs
cp seus_documentos.pdf data/

# Ou criar subpastas
mkdir data/contratos
cp *.pdf data/contratos/
```

### 2. Indexar documentos (Ingestão)

**Usando a CLI (Recomendado):**
```bash
# Ativar ambiente virtual primeiro!
.\.venv\Scripts\Activate.ps1

# Indexar todos os documentos em data/
python main.py ingest

# Opções avançadas:
python main.py ingest --file-types pdf,txt --chunk-size 500 --chunk-overlap 50
```

**Usando Python diretamente:**
```python
from src.ingest import ingest_documents

# Indexar todos os documentos
vectorstore = ingest_documents(
    data_dir='./data',
    persist_dir='./vectorstore',
    file_types=['txt', 'pdf', 'md'],  # Tipos de arquivo
    chunk_size=500,                    # Tamanho dos chunks
    chunk_overlap=50                   # Sobreposição entre chunks
)

print("Indexação concluída!")
```

> **Nota:** A primeira vez que rodar, o sistema baixará o modelo de embeddings (~90MB)

### 3. Consultar o sistema (Queries)

#### Modo Interativo (CLI)
```bash
# Iniciar modo interativo
python main.py query --interactive

# Exemplo de uso:
# > Quais são os cargos do edital?
# > Qual o prazo de validade?
# > exit  (para sair)
```

#### Consulta Única
```bash
# Fazer uma pergunta direta
python main.py query -q "Qual o conteúdo do documento?"

# Com opções personalizadas:
python main.py query -q "Resumo" --top-k 5 --temperature 0.7 --model mistral
```

#### Modo Programático
```python
from src.chain import create_rag_chain
from src.query import RAGQuery

# Criar chain RAG
chain = create_rag_chain(
    vectorstore_path='./vectorstore',
    model_name='llama3',
    top_k=3,
    temperature=0.0
)

# Criar interface de query
query_interface = RAGQuery(chain, model_name='llama3')

# Fazer pergunta
response = query_interface.query("Qual é o assunto principal?")

# Exibir resposta formatada
print(response)

# Ver estatísticas
stats = query_interface.get_stats()
print(f"Total de queries: {stats['total_queries']}")
print(f"Tempo médio: {stats['avg_response_time']:.2f}s")
```

### 4. Adicionar novos documentos

Quando adicionar novos documentos, **re-indexe** para atualizar o vectorstore:

```bash
# 1. Adicionar novos arquivos em data/
cp novo_documento.pdf data/

# 2. Re-indexar
python main.py ingest

# O sistema criará um novo índice com todos os documentos
```

### 5. Ver informações do sistema

```bash
python main.py info
```

Exibe:
- Modelo LLM configurado
- Modelo de embeddings
- Número de documentos indexados
- Localização do vectorstore

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
- [ ] **Sanitização de texto**: Implementar limpeza de documentos (remover caracteres especiais, normalizar Unicode, múltiplos espaços)
- [ ] **Token-based splitting**: Substituir `RecursiveCharacterTextSplitter` por `TokenTextSplitter` para respeitar limites do modelo
- [ ] **Validação de chunks**: Garantir que chunks não excedam 256 tokens do modelo de embeddings
- [ ] Adicionar suporte a mais formatos (DOCX, HTML)
- [ ] Implementar cache de embeddings para evitar reprocessamento
- [ ] Adicionar CLI com argparse
- [ ] Melhorar prompts para casos específicos

### Médio prazo
- [ ] **Pré-processamento avançado**: OCR para PDFs escaneados, limpeza de headers/footers
- [ ] **Modelos de embedding alternativos**: Suporte para modelos multilíngues e otimizados para português
- [ ] **Chunking semântico**: Divisão por seções/parágrafos em vez de apenas tamanho
- [ ] Integrar LangSmith para observabilidade
- [ ] Adicionar avaliação com RAGAS
- [ ] Implementar API REST com FastAPI
- [ ] Adicionar suporte a Vertex AI (produção)

### Longo prazo
- [ ] Interface web (Streamlit/Gradio)
- [ ] Suporte a conversas (chat com memória)
- [ ] Multi-tenancy
- [ ] Deploy com Docker

## 🔍 Troubleshooting

### Problemas Comuns e Soluções

#### 1. Erro: "Microsoft Visual C++ 14.0 or greater is required"
**Problema:** Ao instalar dependências no Windows, falta compilador C++.

**Solução:**
- Instale [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- OU remova `chroma-hnswlib` do requirements.txt (não é obrigatório)

#### 2. Erro: "np.float_ was removed in NumPy 2.0"
**Problema:** Incompatibilidade entre NumPy 2.0+ e sentence-transformers.

**Solução:**
```bash
pip install "numpy==1.26.4" --force-reinstall
```

#### 3. Erro: "Vector store not found"
**Problema:** Tentando fazer query antes de indexar documentos.

**Solução:**
```bash
# Primeiro indexe os documentos
python main.py ingest

# Depois faça queries
python main.py query -q "sua pergunta"
```

#### 4. Erro: "Ollama call failed with status code 404"
**Problema:** Modelo Ollama não está instalado.

**Solução:**
```bash
# Verificar modelos instalados
ollama list

# Instalar modelo necessário
ollama pull llama3
```

#### 5. Erro: "ModuleNotFoundError: No module named 'langchain_community'"
**Problema:** Ambiente virtual não está ativado ou dependências não foram instaladas.

**Solução:**
```bash
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Reinstalar dependências
pip install -r requirements.txt
```

#### 6. Warnings de Deprecation do LangChain
**Problema:** Usando versões antigas de pacotes LangChain.

**Solução:** As versões corretas já estão no `requirements.txt`:
- `langchain-chroma` (não `langchain_community.vectorstores`)
- `langchain-ollama` (não `langchain_community.llms`)
- `langchain-huggingface` (não `langchain_community.embeddings`)

#### 7. ChromaDB Telemetry Errors
**Problema:** Erros de telemetria do ChromaDB no console.

**Solução:** Já configurado no código para desabilitar telemetria automaticamente.

#### 8. Certificado SSL em ambientes corporativos
**Problema:** Erros de certificado ao baixar modelos.

**Solução:**
```bash
# Temporariamente (não recomendado em produção)
set CURL_CA_BUNDLE=
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Dicas de Performance

**Melhorar velocidade de resposta:**
- Use modelos menores: `ollama pull phi` ou `ollama pull mistral`
- Reduza `top_k` para 2 ou 1
- Configure `temperature=0.0` para respostas mais rápidas

**Economizar memória:**
- Use chunks menores: `CHUNK_SIZE=300`
- Processe menos documentos por vez

**Melhorar qualidade das respostas:**
- Aumente `top_k` para 5-7
- Use `chunk_overlap` maior: `100`
- Teste diferentes modelos Ollama

## ❓ FAQ

Perguntas frequentes sobre o projeto? Consulte o **[FAQ.md](FAQ.md)** para:

- Como funciona a sanitização e tokenização?
- Que melhorias implementar na pipeline?
- Troubleshooting de problemas comuns
- Otimização de performance
- Escolha de modelos de embedding

## Recursos Adicionais

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

Este projeto está sob a licença GNU General Public License.

## ✍️ Autor

**Patrick Motin Cruz**
AI Software Developer on IPPUC (Institute for Urban Research and Planning).
Graduate student in Data Science at UTFPR (Federal Technological University of Paraná).
2025

---

**Nota**: Este é um projeto educacional/demonstrativo. Para uso em produção, considere adicionar autenticação, rate limiting, monitoramento e outras funcionalidades enterprise.
