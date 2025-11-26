# Guia de Início Rápido

Configure e execute o **RAG Demo** em menos de 10 minutos.

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter:

- ✅ **Python 3.9+** instalado
- ✅ **Ollama** instalado e rodando
- ✅ **4GB+ RAM** disponível
- ✅ **~2GB espaço em disco** livre

## 🚀 Passo 1: Instalar Ollama e Modelo

### Windows/Mac
1. Baixe o Ollama: [https://ollama.ai](https://ollama.ai)
2. Instale e execute
3. Baixe o modelo Llama 3:

```bash
ollama pull llama3
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
```

### Verificar Instalação
```bash
ollama list
# Deve mostrar: llama3

ollama run llama3 "Hello"
# Deve responder com uma mensagem
```

## 📦 Passo 2: Clonar e Configurar Projeto

```bash
# 1. Clonar repositório
git clone https://github.com/patrickmcruz/rag-demo.git
cd rag-demo

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar ambiente virtual
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# Linux/Mac:
source .venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt
```

**⚠️ Importante**: Sempre ative o ambiente virtual antes de executar comandos!

## 📄 Passo 3: Adicionar Documentos

Coloque seus documentos (PDF, TXT, MD) na pasta `data/`:

```bash
# Criar pasta data (se não existir)
mkdir data

# Copiar seus documentos
cp seus_documentos.pdf data/

# Ou baixar exemplos
# curl -o data/exemplo.pdf https://exemplo.com/documento.pdf
```

**Formatos suportados**:
- ✅ PDF (`.pdf`)
- ✅ Texto (`.txt`)
- ✅ Markdown (`.md`)

## 🔍 Passo 4: Indexar Documentos

Execute a ingestão para processar e indexar seus documentos:

```bash
python main.py ingest
```

**O que acontece**:
1. ⏳ Carrega documentos da pasta `data/`
2. ✂️ Divide em chunks de 500 caracteres
3. 🧠 Gera embeddings (baixa modelo ~90MB na primeira vez)
4. 💾 Salva índice em `./vectorstore/`

**Tempo estimado**: ~10 segundos para 50 páginas

**Saída esperada**:
```
2025-11-26 10:00:00 - INFO - Loading pdf files from ./data
2025-11-26 10:00:02 - INFO - Loaded 52 pdf documents
2025-11-26 10:00:02 - INFO - Created 435 chunks
2025-11-26 10:00:10 - INFO - Successfully ingested 52 documents
```

## 💬 Passo 5: Fazer Perguntas

### Opção A: Query Única

```bash
python main.py query -q "Qual o conteúdo do documento?"
```

### Opção B: Modo Interativo

```bash
python main.py query --interactive
```

Depois digite suas perguntas:
```
> Qual o assunto principal?
Resposta: O documento trata sobre...

> Quais os pontos importantes?
Resposta: Os principais pontos são...

> exit
```

## 🎯 Exemplos Práticos

### Exemplo 1: Consultar Edital

```bash
# Adicionar edital
cp edital_concurso.pdf data/

# Indexar
python main.py ingest

# Consultar
python main.py query -q "Quais são os cargos disponíveis?"
python main.py query -q "Qual o salário?"
python main.py query -q "Como se inscrever?"
```

### Exemplo 2: Analisar Contratos

```bash
# Adicionar contratos
cp contratos/*.pdf data/

# Indexar
python main.py ingest

# Consultar
python main.py query -q "Quais as cláusulas de rescisão?"
python main.py query -q "Qual o prazo de vigência?"
```

### Exemplo 3: Pesquisar Documentação Técnica

```bash
# Adicionar docs
cp documentacao/*.md data/

# Indexar
python main.py ingest

# Consultar
python main.py query -q "Como configurar o sistema?"
python main.py query --interactive
```

## ⚙️ Opções Avançadas

### Customizar Ingestão

```bash
# Apenas PDFs
python main.py ingest --file-types pdf

# Chunks menores
python main.py ingest --chunk-size 300 --chunk-overlap 30

# Múltiplos tipos
python main.py ingest --file-types pdf,txt,md
```

### Customizar Query

```bash
# Recuperar mais documentos
python main.py query -q "pergunta" --top-k 5

# Respostas mais criativas
python main.py query -q "pergunta" --temperature 0.7

# Usar modelo diferente
python main.py --model mistral query -q "pergunta"
```

### Ver Informações do Sistema

```bash
python main.py info
```

## 🔧 Configurações (.env)

Crie/edite o arquivo `.env` para personalizar:

```env
# Modelo LLM
OLLAMA_MODEL=llama3

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Diretórios
DATA_DIR=./data
VECTORSTORE_DIR=./vectorstore

# RAG Config
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_DOCUMENTS=3
TEMPERATURE=0.0

# Desabilitar telemetria
ANONYMIZED_TELEMETRY=False
```

## ❓ Problemas Comuns

### "Vector store not found"
**Solução**: Execute `python main.py ingest` primeiro

### "Ollama call failed with status code 404"
**Solução**: Instale o modelo: `ollama pull llama3`

### "ModuleNotFoundError"
**Solução**: Ative o ambiente virtual: `.\.venv\Scripts\Activate.ps1`

### "Microsoft Visual C++ 14.0 required" (Windows)
**Solução**: Instale [Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**Mais soluções**: Veja [Troubleshooting Guide](troubleshooting.md)

## 🎓 Próximos Passos

Agora que o sistema está rodando:

1. 📖 Leia o [Guia de Modelos](models.md) para trocar de LLM
2. 🧠 Veja o [Guia de Embeddings](embeddings.md) para otimizar
3. 📚 Consulte o [FAQ](../FAQ.md) para dúvidas comuns
4. 🏗️ Entenda a [Arquitetura](../ARCHITECTURE.md) do sistema

## 💡 Dicas

- ✅ Sempre ative o ambiente virtual
- ✅ Re-indexe após adicionar novos documentos
- ✅ Use `--interactive` para explorar documentos
- ✅ Comece com poucos documentos para testar
- ✅ Ajuste `temperature` conforme necessidade

## 🆘 Precisa de Ajuda?

- 📖 [FAQ Completo](../FAQ.md)
- 🐛 [Troubleshooting](troubleshooting.md)
- 💬 [GitHub Issues](https://github.com/patrickmcruz/rag-demo/issues)
- 📧 Email: patrickmcruz@gmail.com

---

**Pronto!** 🎉 Seu sistema RAG está funcionando. Boas consultas!
