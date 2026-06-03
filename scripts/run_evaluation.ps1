Set-Location $PSScriptRoot\..
Write-Host "Generating 70-question test set (if needed)..."
conda run -n rag python tests/generate_test_cases_70.py
Write-Host "Running full evaluation (RAGAS + ablation + guardrails)..."
conda run -n rag python tests/run_full_evaluation.py
