# ============================================================================
# run_all.ps1 -- Relance tous les algos FL avec une configuration IDENTIQUE
# pour produire une comparaison croisée propre.
#
# Usage :
#   conda activate FL
#   cd Refactored
#   .\run_all.ps1                    # tous les algos
#   .\run_all.ps1 fedavg scaffold    # uniquement ceux listés
#
# Les résultats vont dans <algo>/results/. Pré-requis : avoir décommenté
# [tool.flwr.federations.local-sim] dans chaque pyproject.toml.
# ============================================================================

# ---- Hyperparamètres COMMUNS (identiques pour TOUS les algos) -------------
$COMMON = @(
    "seed=42",
    "num-clients=10",
    "num-server-rounds=30",
    "learning-rate=0.01",
    "batch-size=32",
    "local-epochs=2",
    "momentum=0.0",
    "partitioning=`"noniid`"",
    "dirichlet-alpha=0.3",
    "data-heterogeneity=0",
    "epochs-heterogeneity=0",
    "straggler-sim=0",
    "round-deadline-s=0.0",
    "comm-size-ratio=1.0",
    "sim-model-mb=0.0"
)

# ---- Paramètres SPÉCIFIQUES par algo (laissés à leur défaut sinon) --------
$EXTRA = @{
    "fedavg"   = @()
    "fedprox"  = @("mu=0.01")
    "fednova"  = @()
    "fedsgd"   = @("local-epochs=1")          # FedSGD = 1 seul pas SGD
    "scaffold" = @()
    "qfedavg"  = @("qfedavg-q=1.0", "qfedavg-L=100.0")  # L=1/lr (au lieu de 1.0)
    "fairfed"  = @("fairfed-beta=0.5")
    "befl"     = @("befl-battery-j=5000.0", "befl-death-threshold=0.05")
}

# ---- Liste d'algos à lancer (CLI args ou tous par défaut) -----------------
$ALGOS_ALL = @("fedavg", "fedprox", "fednova", "fedsgd", "scaffold",
               "qfedavg", "fairfed", "befl")
$algos = if ($args.Count -gt 0) { $args } else { $ALGOS_ALL }

# ---- Boucle ---------------------------------------------------------------
$flwr = "C:\Users\hp\anaconda3\envs\FL\Scripts\flwr.exe"
$ROOT = $PSScriptRoot
$logDir = Join-Path $ROOT "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($algo in $algos) {
    $appDir = Join-Path $ROOT $algo
    if (-not (Test-Path $appDir)) {
        Write-Host "[SKIP] $algo : dossier absent ($appDir)" -ForegroundColor Yellow
        continue
    }

    # Concatène les overrides (commun + spécifique) en UN seul --run-config
    $overrides = $COMMON + $EXTRA[$algo]
    $cfgArg = ($overrides -join " ")

    $log = Join-Path $logDir "$algo.log"
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ">>> $algo  (log : $log)" -ForegroundColor Cyan
    Write-Host "    overrides : $cfgArg" -ForegroundColor DarkGray
    Write-Host "==========================================================" -ForegroundColor Cyan

    $t0 = Get-Date
    & $flwr run $appDir --run-config $cfgArg 2>&1 | Tee-Object -FilePath $log
    $dt = (Get-Date) - $t0

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $algo en $($dt.TotalSeconds.ToString('F1'))s" -ForegroundColor Red
    } else {
        Write-Host "[OK]   $algo en $($dt.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
    }
}

# ---- Plot final -----------------------------------------------------------
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ">>> Generation des plots de comparaison" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
$py = "C:\Users\hp\anaconda3\envs\FL\python.exe"
& $py (Join-Path $ROOT "plot_results.py")
Write-Host "[done] Plots dans : $(Join-Path $ROOT 'plots')" -ForegroundColor Green
