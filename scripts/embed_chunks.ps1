# Resumable chunk embedding (Nomic). Requires conda env `rag`.
# Usage: conda activate rag   (once per shell)
#        .\scripts\embed_chunks.ps1

Set-Location $PSScriptRoot\..

# Non-interactive shells (CI, background jobs) need the conda hook once per session.
if (-not $env:CONDA_DEFAULT_ENV) {
    $condaHook = Join-Path (conda info --base) "shell\condabin\conda-hook.ps1"
    if (Test-Path $condaHook) {
        . $condaHook
    }
}

conda activate rag
if ($LASTEXITCODE -ne 0 -or $env:CONDA_DEFAULT_ENV -ne "rag") {
    Write-Error "Failed to activate conda env 'rag'. Open a shell and run: conda activate rag"
    exit 1
}

if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "1" }
if (-not $env:EMBEDDING_BATCH) { $env:EMBEDDING_BATCH = "8" }
if (-not $env:EMBED_SAVE_EVERY) { $env:EMBED_SAVE_EVERY = "100" }

Write-Host "Embedding chunks in env '$env:CONDA_DEFAULT_ENV' (resumes from output/regulation_embeddings.json) ..."
python -u -m ingestion.embed_chunks
