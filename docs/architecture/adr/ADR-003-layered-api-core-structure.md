# ADR-003 - Layered API and Core Structure

## Status

Accepted

## Context

O projeto tinha arquivos com nomes genericos e sobrepostos, como `app.py` em mais de um contexto. Isso gerava ambiguidade e acoplava a camada HTTP a detalhes de dominio.

## Decision

A estrutura do codigo sera separada em:
- `src/api/` para camada HTTP
- `src/core/` para regras de negocio e orquestracao RAG

## Consequences

### Positive

- responsabilidades mais claras
- nomenclatura menos ambigua
- testes e manutencao mais simples
- melhor base para evolucao futura

### Negative

- exige refatoracao de imports e documentacao
- adiciona mais arquivos e pastas ao projeto

## Alternatives Considered

### Manter tudo em `src/` com nomes curtos

Rejeitado por manter ambiguidade e piorar navegacao.

### Separar por tecnologia em vez de responsabilidade

Rejeitado porque a separacao por responsabilidade ajuda mais na leitura e no design do sistema.
