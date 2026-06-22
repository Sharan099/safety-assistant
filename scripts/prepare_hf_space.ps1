# Sync backend artifacts into a local HF Space clone, ready to git push.
#
# Usage:
#   .\scripts\prepare_hf_space.ps1
#   .\scripts\prepare_hf_space.ps1 -TargetDir H:\Passive_safety_assistant
#
# Then in TargetDir:
#   git add -A
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

# backend/ tree
$backendSrc = Join-Path $Root "backend"
$backendDst = Join-Path $TargetDir "backend"
if (Test-Path $backendDst) { Remove-Item -Recurse -Force $backendDst }
Copy-Item -Recurse -Force $backendSrc $backendDst
Write-Host "  copied backend/"

# Verify embeddings count
$embPath = Join-Path $TargetDir "output\regulation_embeddings.json"
$check = conda run -n rag python -c "import json; d=json.load(open(r'$embPath')); print(len(d.get('embeddings',{})))"
Write-Host ""
Write-Host "Embeddings vectors: $check (expect 28341)"

Write-Host ""
Write-Host "=== Next: push to Hugging Face ==="
Write-Host "  cd `"$TargetDir`""
Write-Host "  git add Dockerfile requirements.txt README.md config.py backend output .gitattributes .dockerignore"
Write-Host "  git lfs track output/regulation_chunks.json output/regulation_embeddings.json"
Write-Host "  git add output/regulation_embeddings.json"
Write-Host "  git status"
Write-Host "  git commit -m `"Deploy PSA FastAPI backend for Vercel frontend`""
Write-Host "  git push"
Write-Host ""
Write-Host "Then set Space secrets (see deploy/hf-space/README.md) and Vercel:"
Write-Host "  NEXT_PUBLIC_API_URL=https://sharan099-passive-safety-assistant.hf.space/api/v1"
