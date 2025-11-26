# Guia de Embeddings

Entenda, configure e otimize os embeddings no RAG Demo.

## 📋 O que são Embeddings?

**Embeddings** são representações vetoriais (numéricas) de texto que capturam o significado semântico.

```
Texto: "O gato está no telhado"
                ↓
Embedding: [0.234, -0.891, 0.456, ..., 0.123]
           (vetor de 384 números)
```

**Por que são importantes no RAG?**
- 🔍 Permitem busca semântica (não apenas keywords)
- 📊 Medem similaridade entre textos
- ⚡ Permitem recuperação rápida (busca vetorial)

## 🎯 Modelo Atual: all-MiniLM-L6-v2

### Especificações

```yaml
Nome: sentence-transformers/all-MiniLM-L6-v2
Tamanho: ~90MB
Dimensões: 384
Tipo: Sentence Transformer
Base: MiniLM (Microsoft)
Treinamento: 1 bilhão de pares de sentenças
Licença: Apache 2.0
```

### Características

**✅ Vantagens:**
- Pequeno e rápido (~50 sentenças/segundo em CPU)
- Boa qualidade para uso geral
- Suporte multilíngue (incluindo português)
- Bem documentado e testado
- Gratuito e open source

**⚠️ Limitações:**
- Não especializado (genérico)
- 384 dimensões (vs. 768 de modelos maiores)
- Performance em português não é perfeita

### Benchmark

| Tarefa | Score | Rank |
|--------|-------|------|
| Similaridade Semântica | 78.9% | Top 15% |
| Classificação | 76.2% | Top 20% |
| Clustering | 72.1% | Top 25% |

## 🔄 Como Funcionam no RAG Demo

### 1. Fase de Ingestão

```python
# Código simplificado
from langchain_huggingface import HuggingFaceEmbeddings

# Carregar modelo
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Para cada chunk de documento:
chunk = "Este é um trecho do documento..."
vector = embeddings.embed_query(chunk)
# vector = [0.234, -0.891, ..., 0.123] (384 números)

# Salvar no ChromaDB
vectorstore.add(chunk, vector)
```

### 2. Fase de Query

```python
# Query do usuário
query = "Qual o conteúdo do documento?"

# Embedding da query (mesmo modelo!)
query_vector = embeddings.embed_query(query)

# Buscar chunks similares (cosine similarity)
results = vectorstore.similarity_search(query_vector, k=3)
```

### 3. Cálculo de Similaridade

```python
# Similaridade por cosseno
similarity = cosine_similarity(query_vector, chunk_vector)

# Valores:
# 1.0  = idêntico
# 0.8+ = muito similar
# 0.6+ = similar
# 0.4- = pouco similar
```

## 🎨 Modelos Alternativos

### Comparação de Modelos

| Modelo | Tamanho | Dims | Velocidade | Qualidade | Multilíngue |
|--------|---------|------|------------|-----------|-------------|
| **all-MiniLM-L6-v2** | 90MB | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| all-mpnet-base-v2 | 438MB | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| multilingual-e5-base | 560MB | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| paraphrase-multilingual | 470MB | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 1. all-mpnet-base-v2 (Melhor Qualidade)

**Quando usar**: Qualidade é mais importante que velocidade

```python
# src/ingest.py ou .env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

**Características**:
- 768 dimensões (melhor representação)
- Melhor performance em benchmarks
- 5x mais lento que MiniLM
- ~440MB

### 2. multilingual-e5-base (Melhor Multilíngue)

**Quando usar**: Documentos em múltiplos idiomas ou português predominante

```python
EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

**Características**:
- Treinado em 100+ idiomas
- Excelente para português
- 768 dimensões
- ~560MB

### 3. paraphrase-multilingual-MiniLM-L12-v2 (Balanceado)

**Quando usar**: Meio-termo entre qualidade e velocidade

```python
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Características**:
- 384 dimensões
- Melhor multilíngue que all-MiniLM
- ~120MB
- Bom compromisso

## ⚙️ Trocar Modelo de Embedding

### Opção 1: Via .env

```env
# .env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Opção 2: Via Código

```python
# src/ingest.py - linha ~35
def __init__(
    self,
    embedding_model: str = "all-mpnet-base-v2",  # Alterar aqui
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
```

### Opção 3: Via CLI (Futuro)

```bash
python main.py ingest --embedding-model all-mpnet-base-v2
```

### ⚠️ IMPORTANTE

**Sempre re-indexe** após trocar modelo:

```bash
# 1. Deletar vectorstore antigo
rm -rf vectorstore/

# 2. Re-indexar com novo modelo
python main.py ingest
```

**Por quê?** Embeddings de modelos diferentes são incompatíveis!

## 🔍 Otimização de Performance

### 1. Usar GPU (se disponível)

```python
# src/ingest.py
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}  # ou 'mps' para Mac M1/M2
)
```

**Ganho**: 5-10x mais rápido

### 2. Batch Processing

```python
# Embeddings em lote (mais eficiente)
texts = [chunk1, chunk2, chunk3, ...]
vectors = embeddings.embed_documents(texts)  # Todos de uma vez
```

**Ganho**: 2-3x mais rápido que um por um

### 3. Cache de Embeddings

```python
# Salvar embeddings calculados
import pickle

embeddings_cache = {}
for chunk in chunks:
    if chunk not in embeddings_cache:
        embeddings_cache[chunk] = embeddings.embed_query(chunk)
    vector = embeddings_cache[chunk]
```

**Ganho**: Instantâneo para documentos repetidos

## 📊 Avaliação de Qualidade

### Teste Manual

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Textos similares
text1 = "O cachorro corre no parque"
text2 = "Um cão está correndo no jardim"
text3 = "Python é uma linguagem de programação"

v1 = embeddings.embed_query(text1)
v2 = embeddings.embed_query(text2)
v3 = embeddings.embed_query(text3)

# Calcular similaridade
from numpy import dot
from numpy.linalg import norm

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print(f"text1 <-> text2: {cosine_sim(v1, v2):.3f}")  # ~0.75 (similar)
print(f"text1 <-> text3: {cosine_sim(v1, v3):.3f}")  # ~0.15 (diferente)
```

### Benchmarks Automáticos

Use **RAGAS** (Retrieval-Augmented Generation Assessment):

```bash
pip install ragas

# TODO: Implementar avaliação automática
```

## 🎯 Escolhendo o Modelo Certo

### Casos de Uso

#### 📄 Documentos Gerais (Contratos, Editais)
→ **all-MiniLM-L6-v2** (padrão)
- Rápido e eficiente
- Boa qualidade geral

#### 🌐 Múltiplos Idiomas
→ **multilingual-e5-base**
- Melhor para PT-BR
- Suporta 100+ idiomas

#### 🎓 Documentos Técnicos/Acadêmicos
→ **all-mpnet-base-v2**
- Melhor compreensão contextual
- 768 dimensões

#### ⚡ Alta Performance (Muitos Documentos)
→ **all-MiniLM-L6-v2**
- Mais rápido
- Menor consumo de memória

#### 💰 Domínio Específico (Legal, Médico)
→ **Fine-tune** custom model
- Treinar em dados do domínio
- Máxima qualidade

## 🔮 Próximos Passos

1. **Experimente**: Teste diferentes modelos
2. **Meça**: Compare qualidade das respostas
3. **Otimize**: Use GPU se disponível
4. **Documente**: Anote qual modelo funciona melhor

## 📚 Recursos

- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Models](https://huggingface.co/models?library=sentence-transformers)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

**Dúvidas?** Veja o [FAQ](../FAQ.md) ou abra uma [issue](https://github.com/patrickmcruz/rag-demo/issues).
