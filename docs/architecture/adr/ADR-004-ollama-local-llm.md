# ADR-004 - Local LLM Execution with Ollama

## Status

Accepted

## Context

O sistema precisava de uma solucao de inferencia local para responder consultas RAG sem dependencia obrigatoria de um provedor remoto de LLM.

## Decision

O projeto utilizara Ollama como mecanismo principal de execucao local de modelos de linguagem.

## Consequences

### Positive

- simplicidade de instalacao e operacao
- menor dependencia de servicos externos
- melhor controle sobre dados locais
- facilidade para trocar modelos locais

### Negative

- disponibilidade e performance dependem do ambiente local
- variabilidade de modelos e recursos entre maquinas
- menor escalabilidade do que servicos gerenciados

## Alternatives Considered

### APIs remotas de LLM

Rejeitado como default por custo recorrente, dependencia de rede e requisitos de privacidade.

### Execucao direta via bindings nativos por modelo

Rejeitado por aumentar complexidade operacional e acoplamento com implementacoes especificas.
