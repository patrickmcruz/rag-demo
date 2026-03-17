# Usage Guide

Guia de uso ponta a ponta para consumir a API e executar um fluxo completo de RAG, do startup ao chat com streaming.

## Objetivo

Este fluxo cobre:
- verificar se a API esta no ar
- obter informacoes de configuracao
- iniciar ingestao de documentos
- acompanhar o job de ingestao
- validar readiness
- executar queries sincronicas
- consumir streaming de resposta via SSE

## Premissas

- a API esta rodando em `http://localhost:8000`
- a pasta de documentos esta acessivel ao backend
- o frontend ou cliente HTTP consegue enviar JSON

## Fluxo funcional recomendado

1. chamar `/health`
2. chamar `/info`
3. chamar `/health/ready`
4. se nao estiver pronto, executar `POST /ingest`
5. fazer polling em `/ingest/{job_id}`
6. quando concluir, revalidar `/health/ready`
7. habilitar `POST /query` e `POST /chat/stream`
8. exibir resposta e fontes

## Categorias de ferramentas

Este guia esta separado por tipo de ferramenta:
- Frontend Web com JavaScript ou TypeScript
- Terminal com PowerShell
- Terminal com curl

## 1. Frontend Web

### 1.1 Verificar liveness

```ts
const health = await fetch("http://localhost:8000/health");
const healthData = await health.json();

if (!health.ok) {
  throw new Error("API indisponivel");
}
```

### 1.2 Carregar informacoes da instancia

```ts
const info = await fetch("http://localhost:8000/info");
const infoData = await info.json();
```

Campos uteis para UI:
- `vectorstore_exists`
- `data_dir_exists`
- `data_file_counts`
- `model`
- `embedding_model`

### 1.3 Verificar readiness antes de liberar query e chat

```ts
const readiness = await fetch("http://localhost:8000/health/ready");

if (readiness.status === 503) {
  // mostrar CTA para indexar documentos
}
```

### 1.4 Iniciar ingestao

```ts
const ingestResponse = await fetch("http://localhost:8000/ingest", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    data_dir: "./data",
    vectorstore_dir: "./vectorstore",
    file_types: ["pdf", "txt", "md"],
    chunk_size: 350,
    chunk_overlap: 75,
  }),
});

const ingestData = await ingestResponse.json();
```

### 1.5 Fazer polling do job

```ts
async function pollIngestJob(jobId: string) {
  while (true) {
    const res = await fetch(`http://localhost:8000/ingest/${jobId}`);
    const data = await res.json();

    if (data.status === "done") return data;
    if (data.status === "failed") {
      throw new Error(data.error || "Falha na ingestao");
    }

    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}
```

### 1.6 Executar query sincronica

```ts
const queryResponse = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    question: "Qual o conteudo principal do documento?",
    return_sources: true,
    language: "pt",
    top_k: 5,
    temperature: 0.2,
  }),
});

const queryData = await queryResponse.json();
```

### 1.7 Consumir chat com streaming

```ts
async function streamChat(question: string, onToken: (token: string) => void) {
  const response = await fetch("http://localhost:8000/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      language: "pt",
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error("Falha ao iniciar stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      if (!event.startsWith("data: ")) continue;

      const payload = JSON.parse(event.slice(6));

      if (payload.error) throw new Error(payload.error);
      if (payload.done) return;

      onToken(payload.token);
    }
  }
}
```

## 2. PowerShell

### 2.1 Health

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://localhost:8000/health"
```

### 2.2 Info

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://localhost:8000/info"
```

### 2.3 Readiness

```powershell
try {
  Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/health/ready"
} catch {
  $_.Exception.Response.StatusCode.value__
}
```

### 2.4 Ingestao

```powershell
$ingestBody = @{
  data_dir = "./data"
  vectorstore_dir = "./vectorstore"
  file_types = @("pdf", "txt", "md")
  chunk_size = 350
  chunk_overlap = 75
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ingest" `
  -ContentType "application/json" `
  -Body $ingestBody
```

### 2.5 Polling do job de ingestao

```powershell
$jobId = "SEU_JOB_ID"

do {
  $job = Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/ingest/$jobId"

  $job
  Start-Sleep -Seconds 2
} while ($job.status -in @("queued", "running"))
```

### 2.6 Query sincronica

```powershell
$queryBody = @{
  question = "Qual o conteudo principal do documento?"
  return_sources = $true
  language = "pt"
  top_k = 5
  temperature = 0.2
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/query" `
  -ContentType "application/json" `
  -Body $queryBody
```

### 2.7 Chat com streaming

Observacao: `Invoke-RestMethod` nao e a melhor ferramenta para consumir SSE token a token. Para testes rapidos de streaming no PowerShell, prefira:
- `curl.exe`
- um cliente frontend com `fetch`
- script customizado com `HttpClient` e leitura de stream

## 3. curl

### 3.1 Health

```bash
curl http://localhost:8000/health
```

### 3.2 Info

```bash
curl http://localhost:8000/info
```

### 3.3 Readiness

```bash
curl http://localhost:8000/health/ready
```

### 3.4 Ingestao

```bash
curl -X POST "http://localhost:8000/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{\"data_dir\":\"./data\",\"vectorstore_dir\":\"./vectorstore\",\"file_types\":[\"pdf\",\"txt\",\"md\"],\"chunk_size\":350,\"chunk_overlap\":75}"
```

### 3.5 Polling do job de ingestao

```bash
curl http://localhost:8000/ingest/SEU_JOB_ID
```

### 3.6 Query sincronica

```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Qual o conteudo principal do documento?\",\"return_sources\":true,\"language\":\"pt\",\"top_k\":5,\"temperature\":0.2}"
```

### 3.7 Chat com streaming

```bash
curl -N -X POST "http://localhost:8000/chat/stream" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Resuma o documento\",\"language\":\"pt\"}"
```

## Contratos importantes

### Estados do job de ingestao

- `queued`
- `running`
- `done`
- `failed`

### Resposta de query

Campos principais:
- `answer`
- `sources`
- `query`
- `response_time`
- `model_name`

### Evento SSE do chat

Formato:

```text
data: {"token":"texto","done":false}

data: {"token":"","done":true}
```

## Tratamento recomendado no frontend

- tratar `503` como estado de negocio: vectorstore ainda nao pronto
- tratar `409` em `/ingest` como ingestao em andamento
- desabilitar query e chat ate readiness `200`
- guardar `job_id` para retomar polling
- exibir fontes retornadas por `/query`
- exibir estado de streaming durante `/chat/stream`

## Service layer simplificado

```ts
const API_BASE = "http://localhost:8000";

export async function getHealth() {
  return fetch(`${API_BASE}/health`).then((r) => r.json());
}

export async function getInfo() {
  return fetch(`${API_BASE}/info`).then((r) => r.json());
}

export async function getReadiness() {
  return fetch(`${API_BASE}/health/ready`);
}

export async function startIngest(body: Record<string, unknown>) {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return {
    status: res.status,
    data: await res.json(),
  };
}

export async function getIngestStatus(jobId: string) {
  return fetch(`${API_BASE}/ingest/${jobId}`).then((r) => r.json());
}

export async function runQuery(body: Record<string, unknown>) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}
```
