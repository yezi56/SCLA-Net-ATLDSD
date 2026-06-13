$ErrorActionPreference = "Stop"
$root = "D:\Code\ATLDSD"
$env:PYTHONPATH = "$root\src;$root\src\models\deeplabv3plus;$root\src\modules"
$python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$out = "$root\outputs\atldsd\clcs_disease_balanced_deeplabv3plus_efficientnet_b4_150"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$cmd = @(
    "-m", "atldsd_seg.engine.train_clcs",
    "--cuda", "true",
    "--seed", "11",
    "--vocdevkit-path", "D:\dataset\ATLDSD\VOCdevkit",
    "--backbone", "efficientnet_b4",
    "--pretrained", "true",
    "--downsample-factor", "16",
    "--input-shape", "256", "256",
    "--init-epoch", "0",
    "--freeze-epoch", "50",
    "--unfreeze-epoch", "150",
    "--freeze-batch-size", "4",
    "--unfreeze-batch-size", "2",
    "--freeze-train", "true",
    "--init-lr", "0.001",
    "--lr-decay-type", "cos",
    "--save-period", "10",
    "--eval-period", "10",
    "--num-workers", "0",
    "--disease-class-weights", "1.0", "1.5", "1.5", "2.0",
    "--save-dir", "$out\weights",
    "--log-dir", "$out\logs"
)

Set-Content -LiteralPath "$out\train_command.txt" -Value "$python $($cmd -join ' ')" -Encoding UTF8
$process = Start-Process -FilePath $python -ArgumentList $cmd -WorkingDirectory $root -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log" -PassThru -WindowStyle Hidden
Set-Content -LiteralPath "$out\train_pid.txt" -Value $process.Id -Encoding ASCII
Write-Host "Started CLCS disease-balanced Ours-B2 training. PID=$($process.Id)"
Write-Host "Output: $out"
