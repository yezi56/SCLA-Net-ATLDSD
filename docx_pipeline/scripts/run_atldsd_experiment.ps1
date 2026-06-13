param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("deeplabv3plus_mobilenet_150", "deeplabv3plus_efficientnet_b4_150", "deeplabv3plus_mobilenetv3_large_150")]
  [string]$Experiment,

  [string]$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe",

  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$ArgsList = @(
  "-m",
  "atldsd_seg.engine.launch_deeplabv3plus",
  $Experiment,
  "--python",
  $Python
)

if ($DryRun) {
  $ArgsList += "--dry-run"
}

& $Python @ArgsList
