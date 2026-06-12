param(
    [string]$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe",
    [int[]]$Seeds = @(11, 23),
    [int]$TrainCount = 128,
    [int]$ValCount = 64,
    [int]$Epochs = 24
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Runner = Join-Path $ProjectRoot "scripts\run_fast_deeplabv3plus_screen.ps1"

foreach ($Seed in $Seeds) {
    $ExperimentName = "combo_lgc_lcsf_sp384_repconv_balanced_high_gct_${TrainCount}_${ValCount}_e${Epochs}_s${Seed}"
    powershell -ExecutionPolicy Bypass -File $Runner `
        -ExperimentName $ExperimentName `
        -TrainCount $TrainCount `
        -ValCount $ValCount `
        -Seed $Seed `
        -Epochs $Epochs `
        -FreezeEpoch 3 `
        -EvalPeriod 3 `
        -SavePeriod 6 `
        -InputSize 384 `
        -BatchSize 4 `
        -NumWorkers 0 `
        -OptimizerType adam `
        -InitLr 0.0005 `
        -DecoderConvType repconv `
        -ClsWeights "1.0 1.0 2.0 3.0 3.0 4.0" `
        -PrefixWeights "gray=2,rust=1.5,alternaria=1.2,brown=1.2" `
        -AttentionHighType gct `
        -AttentionDecoderType sp `
        -LesionCrossScaleFusion true `
        -LesionLocalGlobalContext true `
        -Python $Python
}
