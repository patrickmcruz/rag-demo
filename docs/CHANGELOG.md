# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não Lançado]

### Em Desenvolvimento
- API REST para consultas remotas
- Interface web com Streamlit
- Suporte para mais formatos de documento (DOCX, HTML)
- Cache de embeddings para melhor performance

---

## [1.0.0] - 2025-11-26

### 🎉 Lançamento Inicial

Primeira versão estável do sistema RAG Demo com todas as funcionalidades principais implementadas.

### ✨ Adicionado

#### Infraestrutura e Configuração
- Estrutura modular do projeto (`src/`, `tests/`, `docs/`)
- Ambiente virtual Python com dependências gerenciadas
- Arquivo `.env` para configurações
- Sistema de logging estruturado
- Suporte multiplataforma (Windows, Linux, macOS)

#### Pipeline de Ingestão (`src/ingest.py`)
- Carregamento de documentos (PDF, TXT, Markdown)
- Splitting inteligente de texto com RecursiveCharacterTextSplitter
- Embeddings com HuggingFace (all-MiniLM-L6-v2)
- Indexação persistente com ChromaDB
- Validações e tratamento de erros robusto
- Logging detalhado do processo

#### RAG Chain (`src/chain.py`)
- Configuração flexível da chain RAG
- Suporte para modelos Ollama (Llama 3, Mistral, Phi, etc.)
- Retriever configurável (top-k, similarity search)
- Prompts otimizados em português e inglês
- Validação de vectorstore antes de queries

#### Interface de Query (`src/query.py`)
- CLI interativa para consultas
- Modo de consulta única
- Respostas estruturadas com metadados
- Rastreamento de fontes dos documentos
- Métricas de performance (tempo de resposta)
- Histórico de consultas
- Estatísticas agregadas

#### CLI Principal (`main.py`)
- Comando `ingest` para indexação de documentos
- Comando `query` para consultas (interativo ou único)
- Comando `info` para informações do sistema
- Argumentos configuráveis (model, temperature, top-k, etc.)

#### Documentação Completa
- **README.md**: Documentação principal com guia completo
- **docs/FAQ.md**: Perguntas frequentes sobre modelos, embeddings e troubleshooting
- **docs/ARCHITECTURE.md**: Arquitetura técnica do sistema
- **docs/guides/**: Guias práticos detalhados
  - Início rápido
  - Configuração de modelos
  - Guia de embeddings
  - Resolução de problemas

#### Testes
- Testes unitários para componentes principais
- Testes de integração da pipeline completa
- Cobertura de código com pytest-cov

### 🔧 Configurações

#### Dependências Principais
- **LangChain** 1.1.0 com componentes atualizados:
  - `langchain-chroma` >= 0.1.0
  - `langchain-ollama` 1.0.0
  - `langchain-huggingface` >= 0.1.0
- **ChromaDB** >= 0.5.0
- **sentence-transformers** 2.3.1
- **NumPy** 1.26.4 (compatibilidade fixada)
- **Ollama** para LLM local

#### Variáveis de Ambiente
```env
OLLAMA_MODEL=llama3
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTORSTORE_DIR=./vectorstore
DATA_DIR=./data
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_DOCUMENTS=3
TEMPERATURE=0.0
ANONYMIZED_TELEMETRY=False
```

### 🐛 Corrigido

#### Problemas de Compatibilidade
- **NumPy 2.0**: Fixada versão em 1.26.4 para compatibilidade com sentence-transformers
- **ChromaDB telemetry**: Erros de telemetria resolvidos com atualização para >= 0.5.0
- **chroma-hnswlib**: Removida dependência problemática no Windows (não obrigatória)

#### Warnings de Deprecação
- Migração para `langchain-chroma` (substituindo `langchain_community.vectorstores`)
- Migração para `langchain-ollama.OllamaLLM` (substituindo `langchain_community.llms.Ollama`)
- Migração para `langchain-huggingface` (substituindo `langchain_community.embeddings`)

#### Problemas de Path
- Corrigido erro "Vector store not found" com validação de diretório
- Suporte melhorado para paths no Windows (PowerShell)
- Criação automática de diretórios necessários

### 📚 Documentação

#### Guias Criados
- Pré-requisitos detalhados por plataforma
- Instruções de instalação passo a passo
- Troubleshooting com 8+ problemas comuns resolvidos
- FAQ com 20+ perguntas e respostas
- Guias de configuração avançada

#### Exemplos de Uso
```bash
# Ingestão
python main.py ingest

# Query única
python main.py query -q "Qual o conteúdo?"

# Query interativa
python main.py query --interactive

# Com configurações customizadas
python main.py --model mistral query -q "Resumo" --top-k 5
```

### 🚀 Performance

#### Otimizações
- Cache de embeddings model (carregamento único)
- Lazy loading de componentes
- Índice persistente com ChromaDB (evita re-indexação)

#### Métricas
- Tempo médio de query: ~1-3s (dependendo do modelo)
- Ingestão: ~50 documentos em < 10s
- Embedding model: ~90MB (download único)

### 🔐 Segurança e Privacidade

- **100% Local**: Todos os modelos rodam localmente via Ollama
- **Sem envio de dados**: Nenhuma informação enviada para serviços externos
- **Telemetria desabilitada**: ChromaDB telemetry desativada por padrão
- **Documentos privados**: Dados nunca saem da máquina local

### 🛠️ Ferramentas de Desenvolvimento

- **pytest**: Framework de testes
- **black**: Formatação de código
- **flake8**: Linting
- **mypy**: Type checking
- **Git**: Controle de versão com commits estruturados

### 📦 Estrutura do Projeto

```
rag-demo/
├── src/              # Código-fonte
├── docs/             # Documentação
├── tests/            # Testes
├── data/             # Documentos para ingestão
├── vectorstore/      # Índice ChromaDB
├── scripts/          # Scripts auxiliares
├── configs/          # Arquivos de configuração
├── examples/         # Exemplos de uso
├── main.py           # CLI principal
├── requirements.txt  # Dependências
└── README.md         # Documentação principal
```

---

## Tipos de Mudanças

- **Adicionado**: Para novas funcionalidades
- **Alterado**: Para mudanças em funcionalidades existentes
- **Descontinuado**: Para funcionalidades que serão removidas
- **Removido**: Para funcionalidades removidas
- **Corrigido**: Para correções de bugs
- **Segurança**: Para correções de vulnerabilidades

---

## Links

- [Repositório GitHub](https://github.com/patrickmcruz/rag-demo)
- [Issues](https://github.com/patrickmcruz/rag-demo/issues)
- [Documentação](docs/README.md)
