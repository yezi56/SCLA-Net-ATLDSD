param(
    [string]$ExperimentName = "fast_boundary1",
    [string]$SourceVocdevkit = "D:\dataset\ATLDSD\VOCdevkit",
    [string]$FastVocdevkit = "",
    [int]$TrainCount = 64,
    [int]$ValCount = 32,
    [int]$Seed = 11,
    [int]$Epochs = 10,
    [int]$FreezeEpoch = 2,
    [int]$EvalPeriod = 2,
    [int]$SavePeriod = 5,
    [int]$InputSize = 256,
    [int]$BatchSize = 4,
    [int]$NumWorkers = 0,
    [string]$OptimizerType = "adam",
    [double]$InitLr = 0.0005,
    [string]$DecoderConvType = "standard",
    [string]$UsePpm = "false",
    [string]$ClsWeights = "1.0 1.0 2.0 3.0 3.0 4.0",
    [string]$MixMode = "none",
    [double]$MixProb = 0.0,
    [double]$MixupAlpha = 0.4,
    [double]$CutmixAlpha = 1.0,
    [double]$ComponentLesionWeight = 0.4,
    [double]$ComponentBoundaryWeight = 0.2,
    [double]$ComponentCenterWeight = 0.2,
    [string]$PrefixWeights = "",
    [string]$AttentionLowType = "",
    [string]$AttentionHighType = "",
    [string]$AttentionAsppType = "",
    [string]$AttentionDecoderType = "",
    [double]$LesionBoundarySharpenAlpha = 0.25,
    [string]$ComponentHighFrequencyRefinement = "",
    [double]$ComponentHighFrequencyRefinementAlpha = 0.2,
    [string]$ComponentFeedbackRefine = "",
    [double]$ComponentFeedbackRefineAlpha = 0.15,
    [string]$LesionCrossScaleFusion = "",
    [double]$LesionCrossScaleFusionAlpha = 0.5,
    [string]$LesionLocalGlobalContext = "",
    [double]$LesionLocalGlobalContextAlpha = 0.5,
    [string]$SeverityConsistencyLoss = "",
    [double]$SeverityConsistencyWeight = 0.1,
    [string]$SeverityLossType = "l1",
    [string]$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ($FastVocdevkit -eq "") {
    $FastVocdevkit = Join-Path $ProjectRoot "outputs\fast_voc\VOCdevkit_fast_${TrainCount}_${ValCount}_seed${Seed}"
}
$RunRoot = Join-Path $ProjectRoot "outputs\atldsd_fast\$ExperimentName"
$WeightsDir = Join-Path $RunRoot "weights"
$LogsDir = Join-Path $RunRoot "logs"
$ReportDir = Join-Path $RunRoot "reports\best_miou"

New-Item -ItemType Directory -Force -Path $WeightsDir, $LogsDir, $ReportDir | Out-Null

$SubsetArgs = @(
    (Join-Path $ProjectRoot "scripts\make_fast_voc_subset.py"),
    "--source-vocdevkit", "$SourceVocdevkit",
    "--output-vocdevkit", "$FastVocdevkit",
    "--train-count", "$TrainCount",
    "--val-count", "$ValCount",
    "--seed", "$Seed",
    "--force"
)
if ($PrefixWeights -ne "") {
    $SubsetArgs += @("--prefix-weights", "$PrefixWeights")
}

& $Python @SubsetArgs
if ($LASTEXITCODE -ne 0) {
    throw "Fast VOC subset creation failed with exit code $LASTEXITCODE"
}

$env:ATLDSD_PROJECT_ROOT = $ProjectRoot
$env:ATLDSD_VOCDEVKIT_PATH = $FastVocdevkit
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\src\models\deeplabv3plus;$ProjectRoot\src\modules;$env:PYTHONPATH"

$TrainArgs = @(
    (Join-Path $ProjectRoot "src\models\deeplabv3plus\train.py"),
    "--cuda", "true",
    "--seed", "$Seed",
    "--num-classes", "6",
    "--backbone", "mobilenetv3_large",
    "--pretrained", "true",
    "--downsample-factor", "16",
    "--attention-type", "none",
    "--decoder-conv-type", "$DecoderConvType",
    "--use-ppm", "$UsePpm",
    "--input-shape", "$InputSize", "$InputSize",
    "--init-epoch", "0",
    "--freeze-epoch", "$FreezeEpoch",
    "--freeze-batch-size", "$BatchSize",
    "--unfreeze-epoch", "$Epochs",
    "--unfreeze-batch-size", "$BatchSize",
    "--freeze-train", "true",
    "--init-lr", "$InitLr",
    "--optimizer-type", "$OptimizerType",
    "--lr-decay-type", "cos",
    "--save-period", "$SavePeriod",
    "--eval-period", "$EvalPeriod",
    "--dataset-name", "ATLDSD_FAST",
    "--vocdevkit-path", "$FastVocdevkit",
    "--dice-loss", "true",
    "--focal-loss", "false",
    "--mix-mode", "$MixMode",
    "--mix-prob", "$MixProb",
    "--mixup-alpha", "$MixupAlpha",
    "--cutmix-alpha", "$CutmixAlpha",
    "--component-aux", "true",
    "--component-lesion-weight", "$ComponentLesionWeight",
    "--component-boundary-weight", "$ComponentBoundaryWeight",
    "--component-center-weight", "$ComponentCenterWeight",
    "--lesion-boundary-sharpen", "true",
    "--lesion-boundary-sharpen-alpha", "$LesionBoundarySharpenAlpha",
    "--num-workers", "$NumWorkers",
    "--auto-export-report", "true",
    "--report-dir", "$ReportDir",
    "--report-checkpoint", "best_miou",
    "--report-split", "val",
    "--report-fps-interval", "20",
    "--save-dir", "$WeightsDir",
    "--log-dir", "$LogsDir",
    "--class-names", "background", "leaf", "rust", "alternaria_leaf_spot", "gray_spot", "brown_spot"
)

if ($ClsWeights -ne "") {
    $TrainArgs += @("--cls-weights")
    $TrainArgs += $ClsWeights.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}
if ($SeverityConsistencyLoss -ne "") {
    $TrainArgs += @(
        "--severity-consistency-loss", $SeverityConsistencyLoss,
        "--severity-consistency-weight", "$SeverityConsistencyWeight",
        "--severity-loss-type", $SeverityLossType
    )
}
if ($ExtraArgs.Count -gt 0) {
    $TrainArgs += $ExtraArgs
}
if ($AttentionLowType -ne "") {
    $TrainArgs += @("--attention-low-type", $AttentionLowType)
}
if ($AttentionHighType -ne "") {
    $TrainArgs += @("--attention-high-type", $AttentionHighType)
}
if ($AttentionAsppType -ne "") {
    $TrainArgs += @("--attention-aspp-type", $AttentionAsppType)
}
if ($AttentionDecoderType -ne "") {
    $TrainArgs += @("--attention-decoder-type", $AttentionDecoderType)
}
if ($ComponentHighFrequencyRefinement -ne "") {
    $TrainArgs += @(
        "--component-high-frequency-refinement", $ComponentHighFrequencyRefinement,
        "--component-high-frequency-refinement-alpha", "$ComponentHighFrequencyRefinementAlpha"
    )
}
if ($ComponentFeedbackRefine -ne "") {
    $TrainArgs += @(
        "--component-feedback-refine", $ComponentFeedbackRefine,
        "--component-feedback-refine-alpha", "$ComponentFeedbackRefineAlpha"
    )
}
if ($LesionCrossScaleFusion -ne "") {
    $TrainArgs += @(
        "--lesion-cross-scale-fusion", $LesionCrossScaleFusion,
        "--lesion-cross-scale-fusion-alpha", "$LesionCrossScaleFusionAlpha"
    )
}
if ($LesionLocalGlobalContext -ne "") {
    $TrainArgs += @(
        "--lesion-local-global-context", $LesionLocalGlobalContext,
        "--lesion-local-global-context-alpha", "$LesionLocalGlobalContextAlpha"
    )
}

$CommandText = @($Python) + $TrainArgs
$CommandText -join " " | Set-Content -LiteralPath (Join-Path $RunRoot "train_command.txt") -Encoding UTF8
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
