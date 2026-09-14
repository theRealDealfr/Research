
# Hierarchical Pipeline Chain Script
# Waits for extract_stage2_embeddings.py to finish (embeddings cached),
# then trains Stage 2 and runs the full hierarchical evaluation.

$proj = "c:\Users\arath\Documents\Project"
$stage2_cache = Join-Path $proj "cached_embeddings\wavlm_stage2_disorder_symptom.pt"
$full_test_cache = Join-Path $proj "cached_embeddings\wavlm_full_test_hierarchical.pt"
$python = Join-Path $proj ".venv\Scripts\python.exe"

Write-Host "=========================================================="
Write-Host " HIERARCHICAL PIPELINE: WAITING FOR EMBEDDINGS..."
Write-Host "=========================================================="

# Poll until both cache files exist (extraction complete)
$maxWait = 3600  # 1 hour timeout
$waited = 0
while ((-not (Test-Path $stage2_cache)) -or (-not (Test-Path $full_test_cache))) {
    if ($waited -ge $maxWait) {
        Write-Host "TIMEOUT: Embedding extraction did not complete within 1 hour. Exiting."
        exit 1
    }
    Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Embeddings not ready yet. Waiting 30s..."
    Start-Sleep -Seconds 30
    $waited += 30
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " STEP 2: TRAINING STAGE 2 DISORDER CLASSIFIER"
Write-Host "=========================================================="
Write-Host "Both cache files found. Starting Stage 2 training..."
Write-Host ""

& $python (Join-Path $proj "train_stage2_disorder.py") 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: train_stage2_disorder.py failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " STEP 3: END-TO-END HIERARCHICAL EVALUATION"
Write-Host "=========================================================="
Write-Host ""

& $python (Join-Path $proj "evaluate_hierarchical.py") 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: evaluate_hierarchical.py failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " PIPELINE COMPLETE! ALL RESULTS SAVED."
Write-Host "=========================================================="
