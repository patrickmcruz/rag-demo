# Aceleração por GPU

Este guia explica como habilitar e otimizar a aceleração por GPU no RAG Demo.

## 📊 Visão Geral

A aplicação suporta aceleração por GPU para componentes de embeddings, proporcionando melhorias significativas de performance:

- **Embeddings (sentence-transformers)**: 3-5x mais rápido com GPU
- **LLM (Ollama)**: Utiliza GPU automaticamente se disponível
- **ChromaDB**: Não utiliza GPU (apenas armazenamento)

## 🔧 Requisitos

### Hardware
- **GPU NVIDIA** com suporte a CUDA:
  - GeForce GTX 1060 (6GB) ou superior
  - RTX série 20xx/30xx/40xx (recomendado)
  - Quadro/Tesla para workstations
- **VRAM**: Mínimo 4GB, recomendado 6GB+

### Software
- **CUDA Toolkit** 11.8 ou superior
- **Drivers NVIDIA** atualizados (versão 520+ para CUDA 11.8)
- **Python** 3.10-3.12

## 📦 Instalação

### 1. Verificar Compatibilidade

Verifique se sua GPU é compatível:

```bash
# Windows
nvidia-smi

# Deve mostrar informações da GPU e versão do driver
```

### 2. Instalar CUDA Toolkit (se necessário)

Baixe e instale do site oficial da NVIDIA:
- https://developer.nvidia.com/cuda-downloads

**Ou use o instalador do conda:**

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6
```

### 3. Instalar PyTorch com Suporte CUDA

```bash
# Ativar ambiente virtual
.venv\Scripts\activate

# Instalar PyTorch com CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1 (GPUs mais recentes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verificar Instalação

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'Versão CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Saída esperada:**
```
PyTorch: 2.x.x+cu118
CUDA disponível: True
Versão CUDA: 11.8
GPU: NVIDIA GeForce RTX 3060
```

## ⚙️ Configuração

### Habilitar GPU na Aplicação

Edite o arquivo `.env`:

```bash
# GPU Configuration
USE_GPU=true
GPU_DEVICE=0  # ID da GPU (0, 1, 2... para múltiplas GPUs)
```

### Múltiplas GPUs

Se você tem múltiplas GPUs, especifique qual usar:

```bash
# Listar GPUs disponíveis
python -c "import torch; print(f'GPUs disponíveis: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Configurar GPU específica no .env
GPU_DEVICE=1  # Usar segunda GPU
```

## 🚀 Uso

### Ingestão de Documentos com GPU

```bash
# Com GPU habilitada no .env
python main.py ingest

# Logs devem mostrar:
# INFO - Using device: cuda:0
# INFO - GPU available: NVIDIA GeForce RTX 3060, using cuda:0
```

### Consultas com GPU

```bash
python main.py query -q "Sua pergunta aqui"

# Modo interativo
python main.py query --interactive
```

## 📈 Benchmark de Performance

### Embeddings (all-MiniLM-L6-v2)

| Operação | CPU (Intel i7) | GPU (RTX 3060) | Speedup |
|----------|----------------|----------------|---------|
| Ingestão de 100 PDFs | ~180s | ~45s | 4.0x |
| Embedding de 1000 chunks | ~25s | ~6s | 4.2x |
| Query (top-k=3) | ~0.8s | ~0.2s | 4.0x |

### Ollama (LLM)

Ollama detecta GPU automaticamente, não requer configuração adicional:

```bash
# Verificar se Ollama está usando GPU
ollama run llama3 "test"

# Monitorar uso da GPU
nvidia-smi -l 1
```

## 🔍 Monitoramento

### Uso de GPU em Tempo Real

```bash
# Monitor contínuo (atualiza a cada 1 segundo)
nvidia-smi -l 1

# Informações detalhadas
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv
```

### Durante Ingestão/Query

```powershell
# Terminal 1: Executar aplicação
python main.py ingest

# Terminal 2: Monitorar GPU
nvidia-smi -l 1
```

## 🐛 Troubleshooting

### "CUDA not available" (PyTorch instalado mas GPU não detectada)

**Causa:** Versão do PyTorch incompatível com versão do CUDA

**Solução:**
```bash
# Verificar versão CUDA do sistema
nvcc --version

# Reinstalar PyTorch com versão correta
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "RuntimeError: CUDA out of memory"

**Causa:** VRAM insuficiente para batch de embeddings

**Solução:**
1. Reduzir batch size (sentence-transformers usa batch automático)
2. Processar menos documentos por vez
3. Usar modelo de embedding menor:

```bash
# .env
EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2  # Menor que all-MiniLM-L6-v2
```

### GPU não acelera significativamente

**Causa:** Overhead de transferência CPU↔GPU para batches pequenos

**Solução:**
- GPU é mais eficiente para ingestão em lote (muitos documentos)
- Para queries isoladas, a diferença é menor
- Considere usar CPU se processar poucos documentos

### Drivers NVIDIA desatualizados

**Causa:** Driver incompatível com CUDA Toolkit

**Solução:**
```bash
# Verificar versão do driver
nvidia-smi

# Baixar driver atualizado:
# https://www.nvidia.com/Download/index.aspx
```

## 💡 Otimizações Avançadas

### 1. Precision Reduzida (FP16)

Para GPUs com Tensor Cores (RTX série):

```python
# Modificar src/chain.py e src/ingest.py
embedding = HuggingFaceEmbeddings(
    model_name=self.embedding_model,
    model_kwargs={
        'device': self.device,
        'torch_dtype': torch.float16  # FP16 para velocidade
    },
    encode_kwargs={'normalize_embeddings': True}
)
```

### 2. Batch Size Otimizado

```python
# src/ingest.py - adicionar ao criar embeddings
encode_kwargs={
    'normalize_embeddings': True,
    'batch_size': 128  # Aumentar para GPUs potentes
}
```

### 3. Múltiplas GPUs (Data Parallel)

Para datasets muito grandes com múltiplas GPUs:

```python
# Modificar src/chain.py
import torch.nn as nn

if torch.cuda.device_count() > 1:
    embedding_model = nn.DataParallel(embedding_model)
```

## 🔗 Recursos Adicionais

- [PyTorch CUDA Semântica](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Sentence Transformers Performance](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html#performance)
- [Ollama GPU Support](https://github.com/ollama/ollama/blob/main/docs/gpu.md)

## 📊 Comparação: CPU vs GPU

### Quando usar GPU:
✅ Ingestão de grandes volumes de documentos (100+ PDFs)  
✅ Reindexação frequente do vectorstore  
✅ Múltiplas queries simultâneas  
✅ Modelos de embedding grandes (>100M parâmetros)  

### Quando CPU é suficiente:
✅ Queries ocasionais em vectorstore já construído  
✅ Poucos documentos (<50)  
✅ Modelos pequenos (all-MiniLM-L6-v2)  
✅ Ambiente de desenvolvimento/testes  

## 🎯 Checklist Pré-Deploy

Antes de usar GPU em produção:

- [ ] GPU tem VRAM suficiente (6GB+)
- [ ] Drivers NVIDIA atualizados
- [ ] CUDA Toolkit instalado corretamente
- [ ] PyTorch detecta CUDA (`torch.cuda.is_available()` = True)
- [ ] Variáveis `USE_GPU=true` e `GPU_DEVICE=X` configuradas
- [ ] Testes de benchmark realizados
- [ ] Monitoramento de GPU implementado
- [ ] Fallback para CPU configurado

---

**💬 Dúvidas?** Abra uma issue em https://github.com/patrickmcruz/rag-demo/issues
