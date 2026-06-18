Set-Location $PSScriptRoot\..
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "H:\hf_cache" }
$env:OCR_BACKEND = "paddle"
$env:DOCLING_IMAGES_SCALE = "0.75"

Write-Host "Comparing Docling 2.103+ vs PaddleOCR 3.7 on data/UN_R14.pdf ..."
conda run -n rag --no-capture-output python scripts/compare_ocr_pipeline.py @args

Write-Host "Results: output/ocr_compare/ocr_pipeline_comparison.json"
