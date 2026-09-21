# Hierarchical Augmented Pipeline Chain Script
# 1. Waits for extract_stage2_embeddings_augmented.py to finish caching embeddings.
# 2. Trains Stage 2 AAFF classifier on augmented data and evaluates on untouched test set.
# 3. Runs End-to-End Hierarchical Evaluation (Stage 1 + Augmented Stage 2 on 169 untouched test samples).

$proj = "c:\Users\arath\Documents\Project"
$augmented_cache = Join-Path $proj "cached_embeddings\wavlm_stage2_disorder_symptom_augmented.pt"
$full_test_cache = Join-Path $proj "cached_embeddings\wavlm_full_test_hierarchical.pt"
$python = Join-Path $proj ".venv\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " GENDER-STRATIFIED PITCH AUGMENTATION PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Poll until augmented cache file exists
$maxWait = 3600  # 1 hour timeout
$waited = 0
while ((-not (Test-Path $augmented_cache)) -or (-not (Test-Path $full_test_cache))) {
    if ($waited -ge $maxWait) {
        Write-Host "TIMEOUT: Augmented embedding extraction did not complete within 1 hour. Exiting." -ForegroundColor Red
        exit 1
    }
    Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Augmented embeddings extracting ($waited s elapsed). Waiting 30s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    $waited += 30
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " STEP 1: TRAINING STAGE 2 CLASSIFIER (AUGMENTED TRAIN SET)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Augmented cache ready. Starting Stage 2 training..." -ForegroundColor Green
Write-Host ""

& $python (Join-Path $proj "train_stage2_augmented.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: train_stage2_augmented.py failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " STEP 2: END-TO-END HIERARCHICAL EVALUATION (UNTOUCHED TEST)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""

& $python (Join-Path $proj "evaluate_hierarchical_augmented.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: evaluate_hierarchical_augmented.py failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PIPELINE COMPLETE! ALL AUGMENTED RESULTS GENERATED." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
