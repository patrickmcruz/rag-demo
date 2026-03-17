# ADR-005 - ChromaDB as Local Vector Store

## Status

Accepted

## Context

O sistema precisava de persistencia vetorial simples, local e integrada ao fluxo RAG sem exigir infraestrutura adicional obrigatoria.

## Decision

O projeto utilizara ChromaDB como vector store persistente da versao atual.

## Consequences

### Positive

- setup simples
- persistencia local direta
- integracao madura com LangChain
- bom custo-beneficio para ambiente de desenvolvimento e uso local

### Negative

- nao e a melhor opcao para cenarios distribuidos ou multi-tenant
- estrategia local dificulta compartilhamento de estado entre instancias

## Alternatives Considered

### FAISS

Rejeitado como default por privilegiar busca local em memoria/arquivo sem o mesmo modelo de persistencia operacional usado aqui.

### Bancos vetoriais gerenciados

Rejeitado por aumentar custo, dependencia externa e complexidade desnecessaria para a fase atual.
