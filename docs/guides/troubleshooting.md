# Guia de Troubleshooting

Soluções para problemas comuns no RAG Demo.

## 📋 Índice
- [Instalação](#instalação)
- [Ingestão](#ingestão)
- [Queries](#queries)
- [Performance](#performance)
- [Ollama](#ollama)

---

## 🔧 Instalação

### 1. Erro: "Microsoft Visual C++ 14.0 or greater is required"

**Problema**: Ao instalar dependências no Windows, falta compilador C++.

**Solução A** (Recomendada):
```bash
# Instalar Build Tools
# Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Durante instalação, selecione "Desktop development with C++"
```

**Solução B**:
```bash
# Ativar Developer Mode
# Configurações → Para desenvolvedores → Modo de Desenvolvedor
```

**Solução C** (Workaround):
```bash
# Remover chroma-hnswlib do requirements.txt (não é obrigatório)
```

### 2. Erro: "`np.float_` was removed in NumPy 2.0"

**Problema**: Incompatibilidade entre NumPy 2.0+ e sentence-transformers.

**Solução**:
```bash
pip install "numpy==1.26.4" --force-reinstall
```

### 3. Erro: "ModuleNotFoundError: No module named 'langchain_community'"

**Problema**: Ambiente virtual não está ativado ou dependências não foram instaladas.

**Solução**:
```bash
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux/Mac:
source .venv/bin/activate

# Reinstalar dependências
pip install -r requirements.txt
```

### 4. SSL Certificate Errors (Ambientes Corporativos)

**Problema**: Erros de certificado ao baixar modelos.

**Solução**:
```bash
# Temporariamente (não recomendado em produção)
set CURL_CA_BUNDLE=
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## 📄 Ingestão

### 5. Erro: "No documents found in ./data"

**Problema**: Pasta `data/` vazia ou não contém arquivos suportados.

**Solução**:
```bash
# Verificar conteúdo
ls data/

# Adicionar documentos
cp seus_documentos.pdf data/

# Verificar tipos de arquivo suportados
python main.py ingest --file-types pdf,txt,md
```

### 6. Erro ao processar PDFs

**Problema**: PDF corrompido ou protegido.

**Solução**:
```bash
# Verificar integridade do PDF
# Tentar abrir em leitor de PDF

# Remover proteção (se permitido)
# qpdf --decrypt input.pdf output.pdf
```

### 7. Ingestão Lenta

**Problema**: Processamento demora muito.

**Solução**:
```bash
# Reduzir chunk_size
python main.py ingest --chunk-size 300

# Processar menos arquivos por vez
python main.py ingest --file-types pdf  # Apenas PDFs

# Usar GPU (se disponível)
# Automático se GPU detectada
```

---

## 💬 Queries

### 8. Erro: "Vector store not found at: ./vectorstore"

**Problema**: Tentando fazer query antes de indexar documentos.

**Solução**:
```bash
# Primeiro indexe os documentos
python main.py ingest

# Depois faça queries
python main.py query -q "sua pergunta"
```

### 9. Erro: "Ollama call failed with status code 404"

**Problema**: Modelo Ollama não está instalado.

**Solução**:
```bash
# Verificar modelos instalados
ollama list

# Instalar modelo necessário
ollama pull llama3

# Ou usar modelo já instalado
python main.py --model mistral query -q "pergunta"
```

### 10. Respostas de Baixa Qualidade

**Problema**: Respostas imprecisas ou irrelevantes.

**Solução A** - Aumentar contexto:
```bash
python main.py query -q "pergunta" --top-k 5
```

**Solução B** - Trocar modelo:
```bash
ollama pull gemma2
python main.py --model gemma2 query -q "pergunta"
```

**Solução C** - Ajustar temperatura:
```bash
python main.py query -q "pergunta" --temperature 0.7
```

**Solução D** - Re-indexar com chunks diferentes:
```bash
python main.py ingest --chunk-size 700 --chunk-overlap 100
```

### 11. Respostas Muito Lentas

**Problema**: Tempo de resposta > 10s.

**Solução**:
```bash
# Usar modelo mais rápido
ollama pull phi3
python main.py --model phi3 query -q "pergunta"

# Reduzir top-k
python main.py query -q "pergunta" --top-k 1

# Verificar se GPU está sendo usada
ollama ps
```

---

## 🚀 Performance

### 12. Alto Uso de Memória

**Problema**: Sistema consome muita RAM.

**Solução**:
```bash
# Usar modelo menor
ollama pull phi3  # Apenas 2.2GB

# Limpar vectorstore antigo
rm -rf vectorstore/
python main.py ingest

# Reduzir batch size (futuro)
```

### 13. Ollama Não Responde

**Problema**: Ollama travado ou não iniciado.

**Solução**:
```bash
# Verificar se está rodando
ollama list

# Reiniciar Ollama
# Windows: Fechar e reabrir aplicação
# Linux: sudo systemctl restart ollama

# Verificar porta
curl http://localhost:11434/api/tags
```

### 14. Erro de Porta (11434) em Uso

**Problema**: Outra aplicação usando porta do Ollama.

**Solução**:
```bash
# Windows: Verificar porta
netstat -ano | findstr :11434

# Linux/Mac:
lsof -i :11434

# Mudar porta Ollama (não recomendado)
# Ou fechar aplicação conflitante
```

---

## 🤖 Ollama

### 15. "Model not found" ao fazer pull

**Problema**: Modelo não existe ou nome incorreto.

**Solução**:
```bash
# Listar modelos disponíveis
ollama list

# Buscar modelos online
# https://ollama.ai/library

# Usar nome correto
ollama pull llama3  # Não "lama3" ou "llama-3"
```

### 16. Download Interrompido

**Problema**: Download do modelo falhou.

**Solução**:
```bash
# Tentar novamente (retoma automaticamente)
ollama pull llama3

# Verificar espaço em disco
df -h  # Linux/Mac
wmic logicaldisk get size,freespace  # Windows

# Verificar conexão internet
ping ollama.ai
```

### 17. Modelo Corrompido

**Problema**: Modelo baixado está corrompido.

**Solução**:
```bash
# Remover modelo
ollama rm llama3

# Baixar novamente
ollama pull llama3

# Verificar integridade
ollama run llama3 "test"
```

---

## 📊 Diagnóstico

### Coletar Informações de Debug

```bash
# Versões
python --version
pip --version
ollama --version

# Modelos instalados
ollama list

# Status do sistema
python main.py info

# Logs detalhados
export LOG_LEVEL=DEBUG  # Linux/Mac
set LOG_LEVEL=DEBUG  # Windows
python main.py ingest
```

### Testar Componentes

```python
# Testar embeddings
python -c "
from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print('Embeddings OK')
"

# Testar Ollama
python -c "
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model='llama3')
print(llm.invoke('test'))
"

# Testar ChromaDB
python -c "
from langchain_chroma import Chroma
print('ChromaDB OK')
"
```

---

## 🆘 Ainda com Problemas?

### Relatar Problema

Ao abrir uma issue, inclua:

1. **Sistema Operacional**: Windows/Linux/Mac
2. **Versão Python**: `python --version`
3. **Versão Ollama**: `ollama --version`
4. **Comando executado**: Ex: `python main.py ingest`
5. **Erro completo**: Copiar mensagem de erro inteira
6. **Logs**: Executar com `LOG_LEVEL=DEBUG`

### Recursos

- 📖 [FAQ Completo](../FAQ.md)
- 🏗️ [Arquitetura](../ARCHITECTURE.md)
- 💬 [GitHub Issues](https://github.com/patrickmcruz/rag-demo/issues)
- 📧 Email: patrickmcruz@gmail.com

---

**Dica**: A maioria dos problemas se resolve com:
1. ✅ Ativar ambiente virtual
2. ✅ Reinstalar dependências
3. ✅ Verificar Ollama está rodando
4. ✅ Re-indexar documentos
