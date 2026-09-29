# =============================================================================
# run_lora_training.ps1
# WavLM-Large + LoRA + AAFF Training Launcher
#
# Usage:
#   .\run_lora_training.ps1 -Mode check      # Gradient & architecture check only
#   .\run_lora_training.ps1 -Mode dryrun     # 5-sample, 1-epoch sanity dry-run
#   .\run_lora_training.ps1 -Mode full       # Full training (T1 disorder_class)
#   .\run_lora_training.ps1 -Mode full -UnfreezeTopLayers 2  # LoRA + top-2 unfrozen
# =============================================================================

param(
    [ValidateSet("check","dryrun","full")]
    [string]$Mode = "check",

    [int]$UnfreezeTopLayers = 0,

    [string]$TargetColumn = "disorder_class",

    [int]$LoraR = 8,
    [int]$LoraAlpha = 16,
    [float]$LoraDropout = 0.1,
    [string]$TargetModules = "q_proj,v_proj",

    [float]$LoraLR = 1e-4,
    [float]$HeadLR = 5e-4,
    [float]$UnfrozenLR = 1e-5,

    [int]$Epochs = 10,
    [int]$BatchSize = 8,
    [int]$EarlyStoppingPatience = 5,
    [string]$WandbMode = "offline"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    WavLM-Large + LoRA + AAFF Fine-Tuning Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Mode              : $Mode"
Write-Host "  Target Column     : $TargetColumn"
Write-Host "  LoRA r / alpha    : $LoraR / $LoraAlpha"
Write-Host "  Target Modules    : $TargetModules"
Write-Host "  Unfreeze Top Layers: $UnfreezeTopLayers"
Write-Host "  LoRA LR / Head LR : $LoraLR / $HeadLR"
Write-Host ""

switch ($Mode) {

    "check" {
        Write-Host "  [MODE: check] Running gradient & architecture verification only..." -ForegroundColor Yellow
        python "$ScriptDir\train_wavlm_lora_aaff.py" `
            --check_only True `
            --target_column $TargetColumn `
            --lora_r $LoraR `
            --lora_alpha $LoraAlpha `
            --lora_dropout $LoraDropout `
            --target_modules $TargetModules `
            --unfreeze_top_layers $UnfreezeTopLayers `
            --max_train_samples 5 `
            --wandb_mode disabled
    }

    "dryrun" {
        Write-Host "  [MODE: dryrun] 5 samples, 1 epoch sanity run..." -ForegroundColor Yellow
        python "$ScriptDir\train_wavlm_lora_aaff.py" `
            --check_only False `
            --target_column $TargetColumn `
            --lora_r $LoraR `
            --lora_alpha $LoraAlpha `
            --lora_dropout $LoraDropout `
            --target_modules $TargetModules `
            --unfreeze_top_layers $UnfreezeTopLayers `
            --lora_lr $LoraLR `
            --head_lr $HeadLR `
            --unfrozen_lr $UnfrozenLR `
            --epochs 1 `
            --batch_size 4 `
            --gradient_accumulation 1 `
            --max_train_samples 5 `
            --max_eval_samples 5 `
            --early_stopping_patience 1 `
            --test_reload True `
            --wandb_mode disabled
    }

    "full" {
        Write-Host "  [MODE: full] Full training run — this may take several hours." -ForegroundColor Green
        python "$ScriptDir\train_wavlm_lora_aaff.py" `
            --check_only False `
            --target_column $TargetColumn `
            --lora_r $LoraR `
            --lora_alpha $LoraAlpha `
            --lora_dropout $LoraDropout `
            --target_modules $TargetModules `
            --unfreeze_top_layers $UnfreezeTopLayers `
            --lora_lr $LoraLR `
            --head_lr $HeadLR `
            --unfrozen_lr $UnfrozenLR `
            --epochs $Epochs `
            --batch_size $BatchSize `
            --gradient_accumulation 2 `
            --early_stopping_patience $EarlyStoppingPatience `
            --test_reload True `
            --wandb_mode $WandbMode
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  DONE. Exit code 0 - Success." -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "  FAILED. Exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    exit $LASTEXITCODE
}
