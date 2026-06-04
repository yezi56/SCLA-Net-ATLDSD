$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_atldsd_experiment.ps1"

& $Runner -Experiment deeplabv3plus_mobilenetv3_large_150
