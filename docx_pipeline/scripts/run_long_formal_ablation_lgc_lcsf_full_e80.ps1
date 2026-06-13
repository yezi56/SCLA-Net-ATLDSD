param(
    [ValidateSet("LGC", "LGC_LCSF")]
    [string]$Variant = "LGC",
    [string]$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe",
    [int[]]$Seeds = @(11, 23),
    [int]$TrainCount = 1148,
    [int]$ValCount = 246,
    [int]$Epochs = 80,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Runner = Join-Path $ProjectRoot "docx_pipeline\scripts\run_fast_deeplabv3plus_screen.ps1"

foreach ($Seed in $Seeds) {
    if ($Variant -eq "LGC") {
        $ExperimentName = "long_lgc_sp384_formal_full_e${Epochs}_s${Seed}"
        $UseLcsf = "false"
    }
    else {
        $ExperimentName = "long_lgc_lcsf_sp384_formal_full_e${Epochs}_s${Seed}"
        $UseLcsf = "true"
    }

    $RunnerArgs = @{
        ExperimentName = $ExperimentName
        TrainCount = $TrainCount
        ValCount = $ValCount
        Seed = $Seed
        Epochs = $Epochs
        FreezeEpoch = 5
        EvalPeriod = 5
        SavePeriod = 10
        InputSize = 384
        BatchSize = 4
        NumWorkers = 0
        OptimizerType = "adam"
        InitLr = 0.0005
        DecoderConvType = "standard"
        ClsWeights = "1.0 1.0 2.0 3.0 3.0 4.0"
        AttentionDecoderType = "sp"
        LesionCrossScaleFusion = $UseLcsf
        LesionLocalGlobalContext = "true"
        Python = $Python
    }

    if ($DryRun) {
        $argText = $RunnerArgs.GetEnumerator() |
            Sort-Object Name |
            ForEach-Object {
                if ($_.Value -is [array]) {
                    "-$($_.Name) $($_.Value -join ' ')"
                }
                else {
                    "-$($_.Name) $($_.Value)"
                }
            }
        Write-Host "powershell -ExecutionPolicy Bypass -File `"$Runner`" $($argText -join ' ')"
        continue
    }

    & $Runner @RunnerArgs
}
