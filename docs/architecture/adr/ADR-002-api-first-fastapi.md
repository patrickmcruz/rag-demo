# ADR-002 - API-first with FastAPI

## Status

Accepted

## Context

O projeto deixou de ser orientado a CLI e passou a priorizar integracao por HTTP. Era necessario definir uma interface primaria para ingestao, consulta, health checks e streaming de resposta.

## Decision

A interface primaria do sistema sera uma API REST e SSE implementada com FastAPI.

## Consequences

### Positive

- documentacao automatica com Swagger e ReDoc
- contratos bem definidos com Pydantic
- integracao simples com clientes HTTP
- suporte natural a endpoints async e streaming

### Negative

- aumento de responsabilidade operacional em comparacao a uma CLI simples
- necessidade de tratamento de lifecycle, readiness e estado em memoria

## Alternatives Considered

### Manter CLI como interface principal

Rejeitado porque dificultava integracao com outros sistemas e clientes.

### Usar Flask

Rejeitado porque FastAPI oferece melhor integracao com tipagem, schemas e documentacao automatica.
