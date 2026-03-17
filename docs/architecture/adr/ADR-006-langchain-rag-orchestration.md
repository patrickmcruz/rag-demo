# ADR-006 - LangChain for RAG Orchestration

## Status

Accepted

## Context

Era necessario um mecanismo para compor retriever, prompt, modelo e parser de saida de forma extensivel e com integracoes prontas para o ecossistema RAG usado no projeto.

## Decision

O projeto utilizara LangChain como camada de orquestracao da pipeline RAG.

## Consequences

### Positive

- composicao clara de runnables
- integracao pronta com Ollama, Chroma e embeddings HuggingFace
- facilita evolucao de prompts, retrievers e estrategias de query

### Negative

- adiciona dependencia significativa ao projeto
- abstrai detalhes que podem dificultar debug em alguns cenarios

## Alternatives Considered

### Implementacao manual da pipeline

Rejeitado por aumentar codigo de infraestrutura e reduzir velocidade de evolucao.

### Framework diferente de orquestracao

Rejeitado porque LangChain ja atende bem ao ecossistema e ao tipo de composicao usado no projeto.
