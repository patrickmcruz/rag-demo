# FAQ - Perguntas Frequentes

## 📚 Índice

- [Modelos Utilizados](#modelos-utilizados)
- [Sanitização e Tokenização](#sanitização-e-tokenização)
- [Embeddings e Performance](#embeddings-e-performance)
- [Troubleshooting](#troubleshooting)

---

## Modelos Utilizados

### ❓ Qual é o modelo principal (LLM) usado no projeto?

O modelo principal é o **Llama 3**, executado localmente via **Ollama**.

#### Especificações:

```yaml
Modelo: llama3
Provider: Ollama (local)
Custo: Gratuito (100% local)
Privacidade: Total (sem envio de dados)
Configuração: .env → OLLAMA_MODEL=llama3
```

#### Por que Llama 3?

**Vantagens:**
- ✅ **Open source** e gratuito
- ✅ **Execução local** - privacidade total
- ✅ **Ótima qualidade** - comparável a GPT-3.5
- ✅ **Multilíngue** - suporta português bem
- ✅ **Flexível** - vários tamanhos (8B, 70B)
- ✅ **Sem limites de uso** ou custos de API

**Desvantagens:**
- ⚠️ Requer hardware local (GPU recomendada)
- ⚠️ Mais lento que APIs cloud
- ⚠️ Menor context window que GPT-4

#### Modelos alternativos suportados:

O projeto suporta qualquer modelo do Ollama. Para trocar:

```bash
# 1. Baixar modelo alternativo
ollama pull phi3          # Rápido, 3.8GB
ollama pull mistral       # Balanceado, 4.1GB
ollama pull codellama     # Especializado em código, 3.8GB
ollama pull gemma2        # Google, 5.4GB

# 2. Configurar no .env
OLLAMA_MODEL=phi3
```

**Comparação de modelos:**

| Modelo | Tamanho | RAM | Qualidade | Velocidade | Uso Ideal |
|--------|---------|-----|-----------|------------|-----------|
| **llama3:8b** | 4.7GB | 8GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Geral, balanceado |
| phi3 | 3.8GB | 6GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Rápido, eficiente |
| mistral | 4.1GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐ | Bom para instruções |
| codellama | 3.8GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐ | Código, técnico |
| llama3:70b | 39GB | 64GB+ | ⭐⭐⭐⭐⭐ | ⭐ | Máxima qualidade |

#### Como usar modelos cloud (produção):

Para ambientes de produção, você pode integrar APIs:

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet")

# Google Vertex AI
from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(model="gemini-pro")
```

---

### ❓ Qual modelo de embedding é usado? É otimizado para português?

O modelo de embedding padrão é o **all-MiniLM-L6-v2**, que é otimizado para **inglês**.

#### Especificações:

```yaml
Modelo: sentence-transformers/all-MiniLM-L6-v2
Base: BERT (Microsoft)
Idioma: Inglês (EN)
Dimensões: 384
Tamanho: ~80MB
Max tokens: 256
Performance: Rápido e eficiente
```

#### ⚠️ Limitação importante:

**Não é otimizado para português!** O modelo foi treinado principalmente em inglês, o que pode impactar:
- Qualidade dos embeddings para textos em PT-BR
- Similaridade semântica entre documentos
- Precisão do retrieval

#### ✅ Modelos recomendados para português:

**1. NeuralMind BERT (melhor para PT-BR):**
```python
EMBEDDING_MODEL=neuralmind/bert-base-portuguese-cased

# Características:
# - Treinado especificamente em português brasileiro
# - 768 dimensões (maior precisão)
# - ~410MB
# - Melhor desempenho em textos PT-BR
```

**2. Multilingual MiniLM (bom compromisso):**
```python
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Características:
# - Suporta 50+ idiomas incluindo português
# - 384 dimensões
# - ~420MB
# - Bom para projetos multilíngues
```

**3. mBERT (multilíngue):**
```python
EMBEDDING_MODEL=bert-base-multilingual-cased

# Características:
# - Suporta 104 idiomas
# - 768 dimensões
# - ~680MB
# - Google, bem estabelecido
```

#### Comparação detalhada:

| Modelo | Idioma | Dimensões | Tamanho | Qualidade PT-BR | Velocidade |
|--------|--------|-----------|---------|-----------------|------------|
| **all-MiniLM-L6-v2** | EN | 384 | 80MB | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **neuralmind/bert-base-portuguese-cased** | PT-BR | 768 | 410MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| paraphrase-multilingual-MiniLM-L12-v2 | Multi | 384 | 420MB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| all-mpnet-base-v2 | EN | 768 | 420MB | ⭐⭐⭐ | ⭐⭐⭐ |
| mBERT | Multi | 768 | 680MB | ⭐⭐⭐ | ⭐⭐ |

#### Como trocar o modelo de embedding:

**1. Atualizar .env:**
```bash
EMBEDDING_MODEL=neuralmind/bert-base-portuguese-cased
```

**2. Reindexar documentos:**
```bash
# Deletar vectorstore antigo
rm -rf vectorstore/

# Reindexar com novo modelo
python main.py ingest
```

**3. Atualizar código (se necessário):**
```python
# src/ingest.py e src/chain.py já suportam
# Basta mudar a variável de ambiente
ingestor = DocumentIngestor(
    embedding_model="neuralmind/bert-base-portuguese-cased"
)
```

#### 💡 Recomendação para documentos em português:

Para **melhor qualidade** em português brasileiro:

```bash
# .env
OLLAMA_MODEL=llama3                                    # LLM: Llama 3 (suporta PT-BR)
EMBEDDING_MODEL=neuralmind/bert-base-portuguese-cased  # Embeddings: otimizado PT-BR
```

**Benefícios esperados:**
- ✅ Melhor compreensão semântica em português
- ✅ Retrieval mais preciso
- ✅ Respostas mais relevantes
- ✅ Menos "perdas" na tradução de conceitos

**Trade-off:**
- ⚠️ Modelo maior (410MB vs 80MB)
- ⚠️ ~2-3x mais lento na indexação
- ⚠️ Mais uso de memória RAM

#### Testando diferentes modelos:

```python
# Script de comparação
from src.ingest import DocumentIngestor
import time

models = [
    "all-MiniLM-L6-v2",
    "neuralmind/bert-base-portuguese-cased",
    "paraphrase-multilingual-MiniLM-L12-v2"
]

for model in models:
    print(f"\nTestando: {model}")
    start = time.time()
    
    ingestor = DocumentIngestor(embedding_model=model)
    # ... indexar documentos ...
    
    print(f"Tempo: {time.time() - start:.2f}s")
```

---

## Sanitização e Tokenização

### ❓ Este projeto faz sanitização dos dados antes de gerar os embeddings?

**Não**, atualmente o projeto **não realiza sanitização explícita** dos dados antes de gerar embeddings. O fluxo é direto:

```
Documento → Loader → Split → Embedding → Chroma
```

As únicas "limpezas" que acontecem são:
- **`.strip()`** nas queries do usuário (para remover espaços em branco)
- **Nada nos documentos originais** - o texto é usado como está

#### O que os loaders fazem:

1. **TextLoader**: Lê arquivo texto bruto sem processamento
2. **PyPDFLoader**: Extrai texto do PDF (pode incluir caracteres especiais, quebras de linha estranhas)
3. **UnstructuredMarkdownLoader**: Processa Markdown básico

#### ⚠️ Problemas potenciais sem sanitização:

- Múltiplos espaços em branco consecutivos
- Caracteres especiais/Unicode mal formados
- Headers/footers repetitivos de PDFs
- Formatação inconsistente entre documentos
- Metadados ou "lixo" de documentos digitalizados

#### ✅ Melhorias recomendadas:

Adicionar uma função de sanitização no pipeline de ingestão:

```python
import re
import unicodedata

def sanitize_text(text: str) -> str:
    """Sanitize text before embedding."""
    # Remove múltiplos espaços
    text = re.sub(r'\s+', ' ', text)
    
    # Remove caracteres de controle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normaliza Unicode (NFKC = compatibilidade)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove linhas vazias múltiplas
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()
```

---

### ❓ Que tipo de tokenização é usada?

A tokenização acontece **dentro do modelo de embeddings**, não explicitamente no código do projeto.

#### Modelo e Tokenizer:

```
Modelo: sentence-transformers/all-MiniLM-L6-v2
Tokenizer: WordPiece (baseado em BERT)
Vocabulário: ~30.000 tokens
Limite: 256 tokens por sequência
```

#### Como funciona atualmente:

1. **RecursiveCharacterTextSplitter** divide por **caracteres**:
   ```python
   separators=["\n\n", "\n", " ", ""]  # Não é tokenização!
   chunk_size=500  # 500 caracteres, não tokens
   chunk_overlap=50  # 50 caracteres de overlap
   ```

2. **HuggingFaceEmbeddings** tokeniza internamente:
   - Usa o tokenizer WordPiece do modelo
   - Trunca automaticamente para 256 tokens se necessário
   - Adiciona tokens especiais: `[CLS]` (início) e `[SEP]` (fim)

#### ⚠️ Problema identificado:

O split é feito por **caracteres** (500), mas o limite do modelo é **256 tokens**. 

- Um chunk de 500 caracteres pode ter ~100-150 tokens (depende do idioma)
- Não há garantia de que todos os chunks caibam no limite do modelo
- Chunks muito longos são truncados silenciosamente

#### ✅ Melhorias recomendadas:

**1. Usar TokenTextSplitter** (divide por tokens, não caracteres):

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=200,      # tokens, não caracteres
    chunk_overlap=20,    # overlap em tokens
    encoding_name="cl100k_base"  # ou use o tokenizer do modelo
)
```

**2. Validar tamanho dos chunks**:

```python
def validate_chunk(chunk: str, max_tokens: int = 256) -> bool:
    """Validate chunk size in tokens."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    tokens = tokenizer.encode(chunk)
    
    if len(tokens) > max_tokens:
        logger.warning(
            f"Chunk exceeds {max_tokens} tokens: {len(tokens)} tokens"
        )
        return False
    
    return True
```

**3. Splitting baseado no tokenizer do modelo**:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Carregar tokenizer do modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Usar função de contagem de tokens
def token_length(text: str) -> int:
    return len(tokenizer.encode(text))

# Configurar splitter com função de tokens
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,          # tamanho em tokens
    chunk_overlap=20,        # overlap em tokens
    length_function=token_length,  # conta tokens, não caracteres
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

---

### ❓ Como melhorar a qualidade dos embeddings?

#### Estratégias recomendadas:

**1. Pré-processamento consistente:**
```python
def preprocess_for_embedding(text: str) -> str:
    """Preprocess text for better embeddings."""
    # Sanitização básica
    text = sanitize_text(text)
    
    # Remover URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remover emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalizar números (opcional)
    # text = re.sub(r'\d+', '<NUM>', text)
    
    return text
```

**2. Chunks semânticos (não apenas por tamanho):**
```python
# Dividir por parágrafos/seções primeiro
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n## ",      # Headers Markdown
        "\n### ",
        "\n\n",       # Parágrafos
        "\n",         # Linhas
        ". ",         # Sentenças
        " ",          # Palavras
        ""
    ],
    chunk_size=200,
    chunk_overlap=20,
    length_function=token_length
)
```

**3. Adicionar metadata relevante:**
```python
# Preservar contexto nos metadados
for chunk in chunks:
    chunk.metadata.update({
        "source": doc.metadata["source"],
        "page": doc.metadata.get("page", 0),
        "section": extract_section_name(chunk.page_content),
        "chunk_index": i,
    })
```

**4. Modelos de embedding alternativos:**
```python
# Para textos em português, considere:
EMBEDDING_MODEL = "neuralmind/bert-base-portuguese-cased"
# ou
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

---

## Embeddings e Performance

### ❓ Por que usar all-MiniLM-L6-v2?

**Vantagens:**
- ✅ Rápido (apenas 6 layers)
- ✅ Leve (~80MB)
- ✅ Bom desempenho geral
- ✅ Funciona offline
- ✅ Sem custos de API

**Desvantagens:**
- ⚠️ Otimizado para inglês
- ⚠️ Limite de 256 tokens
- ⚠️ Menos preciso que modelos maiores

**Alternativas:**

| Modelo | Tamanho | Idioma | Dimensões | Uso |
|--------|---------|--------|-----------|-----|
| `all-MiniLM-L6-v2` | 80MB | EN | 384 | Geral, rápido |
| `paraphrase-multilingual-MiniLM-L12-v2` | 420MB | Multi | 384 | Multilíngue |
| `all-mpnet-base-v2` | 420MB | EN | 768 | Melhor qualidade |
| `neuralmind/bert-base-portuguese-cased` | 410MB | PT-BR | 768 | Português |

---

### ❓ Como otimizar a performance do sistema?

**1. Cache de embeddings:**
```python
# Evitar re-embeddings de documentos já processados
import hashlib

def get_doc_hash(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()

# Verificar se embedding já existe antes de processar
```

**2. Batch processing:**
```python
# Processar múltiplos documentos de uma vez
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding,
    persist_directory=persist_dir,
    batch_size=100  # Ajustar conforme memória
)
```

**3. Configurar Chroma adequadamente:**
```python
# Usar configuração otimizada
from chromadb.config import Settings

chroma_settings = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)
```

---

## Troubleshooting

### ❓ Erro: "Token indices sequence length is longer than the maximum"

**Causa:** Chunks maiores que 256 tokens.

**Solução:** Reduzir `chunk_size` ou usar token-based splitting:

```python
# Opção 1: Reduzir chunk_size
DocumentIngestor(chunk_size=300, chunk_overlap=30)

# Opção 2: Usar TokenTextSplitter
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
```

---

### ❓ PDFs com texto mal formatado

**Causa:** PDFs escaneados ou com formatação complexa.

**Soluções:**

```python
# 1. Usar OCR para PDFs escaneados
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("document.pdf")  # Melhor extração

# 2. Limpar texto extraído
def clean_pdf_text(text: str) -> str:
    # Remove hífens de quebra de linha
    text = re.sub(r'-\n', '', text)
    
    # Remove quebras de linha no meio de palavras
    text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)
    
    # Normaliza espaços
    text = re.sub(r' +', ' ', text)
    
    return text
```

---

### ❓ Respostas genéricas ou imprecisas

**Possíveis causas e soluções:**

**1. Poucos documentos recuperados:**
```python
# Aumentar top_k
chain = create_rag_chain(top_k=5)  # default é 3
```

**2. Chunks muito grandes ou pequenos:**
```python
# Ajustar tamanho ideal (200-300 tokens)
DocumentIngestor(chunk_size=400, chunk_overlap=50)
```

**3. Prompt inadequado:**
```python
# Melhorar prompt no chain.py
template = """Você é um especialista em [DOMÍNIO].
Analise cuidadosamente o contexto fornecido.

Contexto:
{context}

Pergunta: {question}

Instruções:
1. Responda APENAS com informações do contexto
2. Cite trechos relevantes
3. Se não souber, diga claramente

Resposta detalhada:"""
```

**4. Temperatura muito alta:**
```python
# Reduzir temperatura para respostas mais determinísticas
chain = create_rag_chain(temperature=0.0)  # mais factual
```

---

### ❓ Ollama não conecta

**Verificações:**

```powershell
# 1. Verificar se Ollama está rodando
ollama list

# 2. Verificar se modelo existe
ollama pull llama3

# 3. Testar conexão
curl http://localhost:11434/api/tags

# 4. Verificar variável de ambiente
echo $env:OLLAMA_BASE_URL
```

**Configurar .env:**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

## 📚 Referências

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Chroma Vector Database](https://docs.trychroma.com/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

**Última atualização:** 2025  
**Contribuições:** Envie PRs ou abra issues com mais perguntas!
