# Ajustes de Modelo e Recuperacao

Recomendacoes rapidas para melhorar completude das respostas e controlar latencia.

## Configuracao atual (completude 100%)
- EMBEDDING_MODEL: all-MiniLM-L12-v2
- TOP_K_DOCUMENTS: 10
- CHUNK_SIZE / OVERLAP: 350 / 75
- TEMPERATURE: 0.1
- Prompt com instrucoes de completude
- Resultado: lista todos os 6 cargos, latencia ~7-9s

## Opcoes de ajuste

### 1) Equilibrado e mais rapido
- TOP_K_DOCUMENTS: 8
- EMBEDDING_MODEL: all-MiniLM-L12-v2
- Esperado: alta completude, latencia ~10-15% menor

### 2) Qualidade maxima
- TOP_K_DOCUMENTS: 8
- EMBEDDING_MODEL: all-mpnet-base-v2
- Esperado: +qualidade de recuperacao, +~1s ingestao/query

### 3) Manter atual
- TOP_K_DOCUMENTS: 10
- EMBEDDING_MODEL: all-MiniLM-L12-v2
- Esperado: 100% completude, latencia ~7-9s

## Como aplicar
1. Editar `.env` com os valores desejados.
2. Resetar vectorstore se trocar EMBEDDING_MODEL:
   - Remove-Item ./vectorstore -Recurse -Force
3. Re-indexar:
   - python main.py ingest
4. Testar:
   - python main.py query -q "Quais os cargos disponiveis?"

## Prompt (já aplicado em src/chain.py)
- Instrui listar TODOS os itens em formato numerado.
- Pede para indicar "Ver documento para lista completa" se faltar contexto.

## Notas
- TOP_K muito alto (>12) aumenta latencia e ruido.
- Sempre reindexar ao trocar EMBEDDING_MODEL.
- Para mais velocidade sem GPU, reduza batch_size em src/ingest.py (encode_kwargs batch_size=128).
