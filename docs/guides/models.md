# Guia de Modelos LLM

Escolha, configure e otimize modelos de linguagem (LLM) no RAG Demo.

## 📋 O que são LLMs?

**LLM (Large Language Model)** é o modelo que gera as respostas finais baseado no contexto recuperado.

No RAG Demo, usamos **Ollama** para rodar modelos localmente.

## 🎯 Modelo Padrão: Llama 3

### Especificações

```yaml
Nome: llama3 (Meta AI)
Tamanho: ~4.7GB (quantizado 4-bit)
Parâmetros: 8B (8 bilhões)
Context Window: 8192 tokens
Velocidade: ~20-30 tokens/s (CPU), ~100+ tokens/s (GPU)
Licença: Llama 3 Community License
```

### Por que Llama 3?

**✅ Vantagens:**
- Excelente qualidade (comparável a GPT-3.5)
- Suporte multilíngue (bom português)
- Gratuito e open source
- Roda localmente (privacidade total)
- Quantizado 4-bit (economiza memória)

**⚠️ Limitações:**
- Requer hardware moderado (4GB+ RAM)
- Mais lento que APIs cloud
- Context window menor que GPT-4

## 🚀 Modelos Disponíveis no Ollama

### Comparação Completa

| Modelo | Tamanho | Parâmetros | Velocidade | Qualidade | Português |
|--------|---------|------------|------------|-----------|-----------|
| **llama3** | 4.7GB | 8B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| llama3:70b | 40GB | 70B | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| mistral | 4.1GB | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| phi3 | 2.2GB | 3.8B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| gemma2 | 5.4GB | 9B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| qwen2.5 | 4.4GB | 7B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 1. Llama 3 (Recomendado)

**Melhor para**: Uso geral, boa qualidade

```bash
ollama pull llama3
```

**Características**:
- Balanceio perfeito qualidade/velocidade
- Excelente compreensão contextual
- Bom em português
- 8K context window

**Uso no RAG Demo**:
```bash
python main.py --model llama3 query -q "pergunta"
```

### 2. Llama 3:70b (Máxima Qualidade)

**Melhor para**: Tarefas complexas, hardware potente

```bash
ollama pull llama3:70b
```

**Características**:
- Melhor qualidade disponível
- 70 bilhões de parâmetros
- Requer GPU potente
- Mais lento

**Requisitos**: 48GB+ RAM ou GPU com 40GB+ VRAM

### 3. Mistral (Mais Rápido)

**Melhor para**: Respostas rápidas, baixa latência

```bash
ollama pull mistral
```

**Características**:
- Mais rápido que Llama 3
- Boa qualidade
- 4.1GB
- Inglês > Português

### 4. Phi3 (Mais Leve)

**Melhor para**: Hardware limitado, testes rápidos

```bash
ollama pull phi3
```

**Características**:
- Apenas 2.2GB
- Muito rápido
- Qualidade razoável
- Bom para prototipagem

### 5. Gemma2 (Google)

**Melhor para**: Alta qualidade, Google-trained

```bash
ollama pull gemma2
```

**Características**:
- Desenvolvido pelo Google
- Excelente qualidade
- Bom multilíngue
- 9B parâmetros

### 6. Qwen2.5 (Alibaba)

**Melhor para**: Multilíngue, Ásia-focado

```bash
ollama pull qwen2.5
```

**Características**:
- Forte em múltiplos idiomas
- Desenvolvido pela Alibaba
- 7B parâmetros
- Boa qualidade geral

## ⚙️ Trocar Modelo

### Via Linha de Comando

```bash
# Opção 1: Argumento global
python main.py --model mistral query -q "pergunta"

# Opção 2: Modo interativo
python main.py --model phi3 query --interactive
```

### Via Arquivo .env

```env
# .env
OLLAMA_MODEL=mistral
```

Depois execute normalmente:
```bash
python main.py query -q "pergunta"
```

### Via Código

```python
# src/chain.py - linha ~75
llm = OllamaLLM(
    model="mistral",  # Alterar aqui
    temperature=self.temperature,
)
```

## 🎛️ Parâmetros do LLM

### Temperature

Controla aleatoriedade das respostas:

```python
# Factual (recomendado para RAG)
temperature=0.0  # Respostas determinísticas

# Balanceado
temperature=0.3  # Pouca criatividade

# Criativo
temperature=0.7  # Mais variado

# Muito criativo
temperature=1.0  # Máxima aleatoriedade
```

**No RAG Demo**:
```bash
python main.py query -q "pergunta" --temperature 0.7
```

### Top-K (Retrieval)

Número de documentos recuperados:

```bash
# Mais focado
--top-k 1  # Apenas 1 documento

# Padrão (recomendado)
--top-k 3  # 3 documentos

# Mais contexto
--top-k 5  # 5 documentos

# Muito contexto (pode ter ruído)
--top-k 10
```

### Top-P e Top-K (Sampling)

Controla vocabulário na geração:

```python
# Em Ollama (futuro)
llm = OllamaLLM(
    model="llama3",
    temperature=0.7,
    top_k=40,  # Top 40 tokens mais prováveis
    top_p=0.9,  # Nucleus sampling
)
```

## 📊 Benchmarks

### Tempo de Resposta (CPU Intel i7)

| Modelo | Tempo Médio | Tokens/segundo |
|--------|-------------|----------------|
| phi3 | 0.5s | 50 |
| mistral | 1.2s | 25 |
| llama3 | 2.0s | 15 |
| gemma2 | 2.5s | 12 |
| llama3:70b | 15s | 2 |

### Qualidade (Benchmark MMLU)

| Modelo | Score | Rank |
|--------|-------|------|
| llama3:70b | 79.2% | Top 1% |
| gemma2 | 71.3% | Top 5% |
| llama3 | 68.4% | Top 10% |
| qwen2.5 | 65.5% | Top 15% |
| mistral | 60.1% | Top 20% |
| phi3 | 68.8% | Top 10% |

## 🔧 Otimização

### 1. Usar GPU

**Automático**: Ollama usa GPU se disponível

**Verificar**:
```bash
ollama run llama3 --verbose
# Mostra: "Using GPU: NVIDIA RTX 3080"
```

**Ganho**: 5-10x mais rápido

### 2. Quantização

Modelos já vêm quantizados (4-bit), mas você pode escolher:

```bash
# Mais rápido, menor qualidade
ollama pull llama3:q4_0

# Balanceado (padrão)
ollama pull llama3

# Melhor qualidade, mais lento
ollama pull llama3:q8_0

# Sem quantização (muito lento)
ollama pull llama3:fp16
```

### 3. Context Window

Modelos têm limites de context:

| Modelo | Context Window |
|--------|----------------|
| llama3 | 8192 tokens |
| mistral | 8192 tokens |
| gemma2 | 8192 tokens |
| phi3 | 4096 tokens |

**⚠️ Cuidado**: Muito contexto = mais lento

### 4. Streaming

Respostas em tempo real (futuro):

```python
for chunk in llm.stream("pergunta"):
    print(chunk, end="", flush=True)
```

## 🎯 Escolhendo o Modelo

### Por Caso de Uso

#### 📄 Documentos Corporativos
→ **llama3** (balanceado)

#### ⚡ Protótipo Rápido
→ **phi3** (leve e rápido)

#### 🎓 Análise Profunda
→ **llama3:70b** (máxima qualidade)

#### 🌐 Múltiplos Idiomas
→ **qwen2.5** (multilíngue)

#### 💰 Hardware Limitado
→ **phi3** (apenas 2.2GB)

### Por Hardware

#### 💻 CPU (8GB RAM)
→ **phi3** ou **mistral**

#### 💻 CPU (16GB+ RAM)
→ **llama3** ou **gemma2**

#### 🎮 GPU (8GB VRAM)
→ **llama3** ou **mistral**

#### 🎮 GPU (16GB+ VRAM)
→ **gemma2** ou **qwen2.5**

#### 🚀 GPU (40GB+ VRAM)
→ **llama3:70b**

## 📚 Modelos Especializados

### Código (Programação)

```bash
ollama pull codellama
ollama pull deepseek-coder
```

### Matemática

```bash
ollama pull wizardmath
ollama pull llemma
```

### Medicina

```bash
ollama pull meditron
ollama pull biomedlm
```

## 🆘 Troubleshooting

### "Model not found"

```bash
# Listar modelos disponíveis
ollama list

# Baixar modelo
ollama pull llama3
```

### Muito Lento

```bash
# Usar modelo menor
ollama pull phi3

# Ou verificar GPU
nvidia-smi  # Linux
```

### Respostas Ruins

```bash
# Aumentar temperatura
--temperature 0.7

# Ou usar modelo melhor
ollama pull gemma2
```

## 🔮 Próximos Passos

1. **Teste**: Experimente diferentes modelos
2. **Compare**: Veja qual funciona melhor para seu caso
3. **Otimize**: Use GPU se disponível
4. **Documente**: Anote configurações que funcionam

## 📚 Recursos

- [Ollama Models](https://ollama.ai/library)
- [Llama 3 Paper](https://ai.meta.com/llama/)
- [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

**Dúvidas?** Consulte o [FAQ](../FAQ.md) ou abra uma [issue](https://github.com/patrickmcruz/rag-demo/issues).
