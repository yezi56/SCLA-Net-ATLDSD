param(
    [string]$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe",
    [int[]]$Seeds = @(11, 23),
    [int]$TrainCount = 128,
    [int]$ValCount = 64,
    [int]$Epochs = 24,
    [int]$DiceClassStart = 1,
    [string]$PrefixWeights = "gray=2,rust=1.5,alternaria=1.2,brown=1.2",
    [string]$ExperimentTag = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Runner = Join-Path $ProjectRoot "scripts\run_fast_deeplabv3plus_screen.ps1"

foreach ($Seed in $Seeds) {
    $TagPart = if ($ExperimentTag -ne "") { "_$ExperimentTag" } else { "" }
    $ExperimentName = "combo_lgc_lcsf_sp384_repconv_balanced_fgdice${DiceClassStart}${TagPart}_${TrainCount}_${ValCount}_e${Epochs}_s${Seed}"
    $RunnerArgs = @{
        ExperimentName = $ExperimentName
        TrainCount = $TrainCount
        ValCount = $ValCount
        Seed = $Seed
        Epochs = $Epochs
        FreezeEpoch = 3
        EvalPeriod = 3
        SavePeriod = 6
        InputSize = 384
        BatchSize = 4
        NumWorkers = 0
        OptimizerType = "adam"
        InitLr = 0.0005
        DecoderConvType = "repconv"
        ClsWeights = "1.0 1.0 2.0 3.0 3.0 4.0"
        PrefixWeights = $PrefixWeights
        AttentionDecoderType = "sp"
        LesionCrossScaleFusion = "true"
        LesionLocalGlobalContext = "true"
        ExtraArgs = @("--dice-class-start", "$DiceClassStart")
        Python = $Python
    }
    & $Runner @RunnerArgs
}
