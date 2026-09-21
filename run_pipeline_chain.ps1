# Hierarchical Pipeline Chain Script
# Waits for extract_stage2_embeddings.py to finish (embeddings cached),
# optionally runs Bayesian hyperparameter tuning, then trains Stage 2 and runs full hierarchical evaluation.

param(
    [switch]$Tune = $false,
    [int]$TuneTrials = 30
)

$proj = $PSScriptRoot
if (-not $proj) { $proj = Get-Location }

$stage2_cache = Join-Path $proj "cached_embeddings\wavlm_stage2_disorder_symptom.pt"
$full_test_cache = Join-Path $proj "cached_embeddings\wavlm_full_test_hierarchical.pt"

# Detect Python executable: check local .venv first, otherwise fall back to system 'python'
$venv_python = Join-Path $proj ".venv\Scripts\python.exe"
if (Test-Path $venv_python) {
    $python = $venv_python
} else {
    $python = "python"
}

Write-Host "=========================================================="
Write-Host " HIERARCHICAL PIPELINE: WAITING FOR EMBEDDINGS..."
Write-Host " Project Root : $proj"
Write-Host " Python Exec  : $python"
Write-Host " Bayesian Tune: $(if ($Tune) { "Enabled ($TuneTrials trials)" } else { "Disabled (use -Tune to enable)" })"
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
Write-Host "Both cache files found."

# Optional Step: Bayesian Hyperparameter Tuning
if ($Tune) {
    Write-Host ""
    Write-Host "=========================================================="
    Write-Host " STEP 2A: BAYESIAN HYPERPARAMETER TUNING (STAGE 2)"
    Write-Host "=========================================================="
    Write-Host "Optimizing learning rate, weight decay, dropout, and batch size..."
    
    & $python (Join-Path $proj "tune_bayesian_stage2.py") --n_trials $TuneTrials 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: tune_bayesian_stage2.py exited with code $LASTEXITCODE. Falling back to default Stage 2 training."
    } else {
        # Copy best bayesian model as the active best model for hierarchical evaluation
        $bayes_ckpt = Join-Path $proj "results_controlled_baselines\model_stage2_disorder_bayesian_best.pt"
        $target_ckpt = Join-Path $proj "results_controlled_baselines\model_stage2_disorder_best.pt"
        if (Test-Path $bayes_ckpt) {
            Copy-Item $bayes_ckpt -Destination $target_ckpt -Force
            Write-Host "Copied Bayesian-optimized checkpoint to $target_ckpt"
        }
    }
} else {
    Write-Host ""
    Write-Host "=========================================================="
    Write-Host " STEP 2: TRAINING STAGE 2 DISORDER CLASSIFIER"
    Write-Host "=========================================================="
    Write-Host "Starting Stage 2 training with grid search baseline..."
    Write-Host ""

    & $python (Join-Path $proj "train_stage2_disorder.py") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: train_stage2_disorder.py failed with exit code $LASTEXITCODE"
        exit 1
    }
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
