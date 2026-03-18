[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://localhost:8000",
    [string]$DataDir = "./data",
    [string]$VectorstoreDir = "./vectorstore",
    [string]$Question = "Qual o conteudo principal do documento?",
    [string[]]$FileTypes = @("pdf", "txt", "md"),
    [int]$ChunkSize = 350,
    [int]$ChunkOverlap = 75,
    [int]$PollIntervalSeconds = 2,
    [int]$MaxPollAttempts = 180,
    [switch]$SkipIngest,
    [switch]$SkipStreaming
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-JsonRequest {
    param(
        [Parameter(Mandatory = $true)][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [object]$Body
    )

    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $Uri
    }

    $jsonBody = $Body | ConvertTo-Json -Depth 10 -Compress
    return Invoke-RestMethod `
        -Method $Method `
        -Uri $Uri `
        -ContentType "application/json; charset=utf-8" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($jsonBody))
}

function Test-Readiness {
    param([string]$BaseUrl)

    try {
        return Invoke-RestMethod -Method Get -Uri "$BaseUrl/health/ready"
    } catch {
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode.value__ -eq 503) {
            return $null
        }
        throw
    }
}

function Start-Ingestion {
    param(
        [string]$BaseUrl,
        [string]$InputDataDir,
        [string]$InputVectorstoreDir
    )

    $requestBody = @{
        data_dir = $InputDataDir
        vectorstore_dir = $InputVectorstoreDir
        file_types = $FileTypes
        chunk_size = $ChunkSize
        chunk_overlap = $ChunkOverlap
    }

    return Invoke-JsonRequest -Method Post -Uri "$BaseUrl/ingest" -Body $requestBody
}

function Wait-IngestionJob {
    param(
        [string]$BaseUrl,
        [string]$JobId
    )

    for ($attempt = 1; $attempt -le $MaxPollAttempts; $attempt++) {
        $job = Invoke-RestMethod -Method Get -Uri "$BaseUrl/ingest/$JobId"
        $statusLine = "job=$($job.job_id) status=$($job.status)"

        if ($job.started_at) {
            $statusLine += " started_at=$($job.started_at)"
        }

        if ($job.completed_at) {
            $statusLine += " completed_at=$($job.completed_at)"
        }

        Write-Host $statusLine

        switch ($job.status) {
            "done" { return $job }
            "failed" { throw "Ingestion failed: $($job.error)" }
            "queued" { Start-Sleep -Seconds $PollIntervalSeconds }
            "running" { Start-Sleep -Seconds $PollIntervalSeconds }
            default { throw "Unexpected ingestion status: $($job.status)" }
        }
    }

    throw "Timed out waiting for ingestion job after $MaxPollAttempts attempts."
}

function Invoke-StreamingChat {
    param(
        [string]$BaseUrl,
        [string]$Prompt
    )

    $handler = New-Object System.Net.Http.HttpClientHandler
    $client = New-Object System.Net.Http.HttpClient($handler)
    $request = $null
    $content = $null
    $reader = $null
    $stream = $null
    $response = $null

    try {
        $payload = @{
            question = $Prompt
            language = "pt"
        } | ConvertTo-Json -Compress

        $content = New-Object System.Net.Http.StringContent(
            $payload,
            [System.Text.Encoding]::UTF8,
            "application/json"
        )

        $request = New-Object System.Net.Http.HttpRequestMessage(
            [System.Net.Http.HttpMethod]::Post,
            "$BaseUrl/chat/stream"
        )
        $request.Content = $content

        $response = $client.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        ).GetAwaiter().GetResult()

        $response.EnsureSuccessStatusCode() | Out-Null

        $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

        Write-Host "Streaming response:"

        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            if ($line.StartsWith("data: ")) {
                $payloadLine = $line.Substring(6) | ConvertFrom-Json

                if ($payloadLine.error) {
                    throw "Streaming error: $($payloadLine.error)"
                }

                if ($payloadLine.done) {
                    Write-Host ""
                    break
                }

                Write-Host -NoNewline $payloadLine.token
            }
        }
    } finally {
        if ($request) { $request.Dispose() }
        if ($content) { $content.Dispose() }
        if ($reader) { $reader.Dispose() }
        if ($stream) { $stream.Dispose() }
        if ($response) { $response.Dispose() }
        $client.Dispose()
        $handler.Dispose()
    }
}

Write-Step "Checking /health"
$health = Invoke-RestMethod -Method Get -Uri "$ApiBaseUrl/health"
$health | Format-List

Write-Step "Checking /info"
$info = Invoke-RestMethod -Method Get -Uri "$ApiBaseUrl/info"
$info | Format-List

Write-Step "Checking /health/ready"
$ready = Test-Readiness -BaseUrl $ApiBaseUrl
if ($ready) {
    $ready | Format-List
} else {
    Write-Host "API not ready yet. Vectorstore may not exist."
}

if (-not $SkipIngest) {
    Write-Step "Triggering /ingest"
    $ingestAccepted = Start-Ingestion -BaseUrl $ApiBaseUrl -InputDataDir $DataDir -InputVectorstoreDir $VectorstoreDir
    $ingestAccepted | Format-List

    Write-Step "Polling /ingest/{job_id}"
    $finalJob = Wait-IngestionJob -BaseUrl $ApiBaseUrl -JobId $ingestAccepted.job_id
    $finalJob | Format-List
}

Write-Step "Checking /health/ready after ingestion"
$readyAfterIngest = Test-Readiness -BaseUrl $ApiBaseUrl
if (-not $readyAfterIngest) {
    throw "API is still not ready after the happy path ingestion."
}
$readyAfterIngest | Format-List

Write-Step "Running /query"
$queryResponse = Invoke-JsonRequest -Method Post -Uri "$ApiBaseUrl/query" -Body @{
    question = $Question
    return_sources = $true
    language = "pt"
    top_k = 5
    temperature = 0.2
}

$queryResponse | Format-List

if (-not $SkipStreaming) {
    Write-Step "Running /chat/stream"
    Invoke-StreamingChat -BaseUrl $ApiBaseUrl -Prompt $Question
}

Write-Step "Happy path completed"
