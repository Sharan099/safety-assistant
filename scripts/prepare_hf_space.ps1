# Sync backend artifacts into a local HF Space clone, ready to git push.
#
# Usage:
#   conda activate rag
#   .\scripts\prepare_hf_space.ps1
#   .\scripts\prepare_hf_space.ps1 -TargetDir H:\Passive_safety_assistant
#
# Then in TargetDir:
#   git add Dockerfile requirements.txt README.md config.py backend ingestion output .gitattributes .dockerignore
#   git status   # confirm regulation_embeddings.json is LFS
#   git commit -m "Deploy PSA backend API"
#   git push

param(
    [string]$TargetDir = "",
    [string]$RepoUrl = "https://huggingface.co/spaces/sharan099/Passive_safety_assistant"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

if (-not (Test-Path "$Root\config.py")) {
    Write-Error "Could not find project root (config.py) from $Root"
}

if (-not $TargetDir) {
    $TargetDir = Join-Path (Split-Path $Root -Parent) "Passive_safety_assistant"
}

Write-Host "Project root : $Root"
Write-Host "HF Space dir : $TargetDir"
Write-Host ""

# Clone HF Space if missing
if (-not (Test-Path $TargetDir)) {
    Write-Host "Cloning $RepoUrl ..."
    git clone $RepoUrl $TargetDir
}

# Ensure Git LFS for embeddings
Push-Location $TargetDir
git lfs install 2>$null
Pop-Location

$files = @(
    @{ Src = "deploy\hf-space\Dockerfile";           Dst = "Dockerfile" },
    @{ Src = "deploy\hf-space\requirements.txt";    Dst = "requirements.txt" },
    @{ Src = "deploy\hf-space\README.md";           Dst = "README.md" },
    @{ Src = "deploy\hf-space\.gitattributes";      Dst = ".gitattributes" },
    @{ Src = "deploy\hf-space\.dockerignore";       Dst = ".dockerignore" },
    @{ Src = "deploy\hf-space\.gitignore";          Dst = ".gitignore" },
    @{ Src = "config.py";                           Dst = "config.py" },
    @{ Src = "output\regulation_chunks.json";       Dst = "output\regulation_chunks.json" },
    @{ Src = "output\regulation_embeddings.json";   Dst = "output\regulation_embeddings.json" },
    @{ Src = "output\ingest_manifest.json";         Dst = "output\ingest_manifest.json" }
)

foreach ($f in $files) {
    $src = Join-Path $Root $f.Src
    $dst = Join-Path $TargetDir $f.Dst
    if (-not (Test-Path $src)) {
        Write-Error "Missing required file: $src"
    }
    $dstParent = Split-Path $dst -Parent
    if ($dstParent -and -not (Test-Path $dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }
    Copy-Item -Force $src $dst
    Write-Host "  copied $($f.Dst)"
}

# backend/ and ingestion/ trees (upload API + document ingest)
foreach ($tree in @("backend", "ingestion")) {
    $treeSrc = Join-Path $Root $tree
    $treeDst = Join-Path $TargetDir $tree
    if (-not (Test-Path $treeSrc)) {
        Write-Error "Missing required folder: $treeSrc"
    }
    if (Test-Path $treeDst) { Remove-Item -Recurse -Force $treeDst }
    Copy-Item -Recurse -Force $treeSrc $treeDst
    Write-Host "  copied $tree/"
}

# Verify embeddings count
$embPath = Join-Path $TargetDir "output\regulation_embeddings.json"
$check = python -c "import json; d=json.load(open(r'$embPath',encoding='utf-8')); print(len(d.get('embeddings',{})))"
Write-Host ""
Write-Host "Embeddings vectors: $check (expect 14554)"

Write-Host ""
Write-Host "=== Next: push to Hugging Face ==="
Write-Host "  cd `"$TargetDir`""
Write-Host "  git lfs track output/regulation_embeddings.json"
Write-Host "  git add Dockerfile requirements.txt README.md config.py backend ingestion output .gitattributes .dockerignore"
Write-Host "  git status"
Write-Host "  git commit -m `"Deploy PSA FastAPI backend (v4 corpus + metadata filtering)`""
Write-Host "  git push"
Write-Host ""
Write-Host "Then set Space secrets (see deploy/hf-space/README.md) and Vercel:"
Write-Host "  NEXT_PUBLIC_API_URL=/api/v1"
Write-Host "  NEXT_PUBLIC_HF_BACKEND_URL=https://sharan099-passive-safety-assistant.hf.space"
