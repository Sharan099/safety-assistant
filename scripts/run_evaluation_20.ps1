Set-Location $PSScriptRoot\..
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:EVAL_SKIP_LLM = "false"
$env:EVAL_TEST_CASES = "tests/test_cases_20.json"
$env:EVAL_RESULTS_NAME = "rag_eval_20_results.json"
$env:EVAL_PNG_SUFFIX = "_20"

Write-Host "Generating 20-question test set..."
& "C:\Users\HP\anaconda3\envs\rag\python.exe" tests/generate_test_cases_20.py

Write-Host "Running evaluation with Groq LLM (20 questions)..."
& "C:\Users\HP\anaconda3\envs\rag\python.exe" tests/run_full_evaluation.py

Write-Host "Done. Results: output/evaluation/rag_eval_20_results.json"
