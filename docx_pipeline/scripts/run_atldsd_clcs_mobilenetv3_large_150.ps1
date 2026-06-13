$ErrorActionPreference = "Stop"
$root = "D:\Code\ATLDSD"
$env:PYTHONPATH = "$root\src;$root\src\models\deeplabv3plus;$root\src\modules"
$python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$out = "$root\outputs\atldsd\clcs_deeplabv3plus_mobilenetv3_large_150"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$cmd = @(
    "-m", "atldsd_seg.engine.train_clcs",
    "--cuda", "true",
    "--seed", "11",
    "--vocdevkit-path", "D:\dataset\ATLDSD\VOCdevkit",
    "--backbone", "mobilenetv3_large",
    "--pretrained", "true",
    "--downsample-factor", "16",
    "--input-shape", "256", "256",
    "--init-epoch", "0",
    "--freeze-epoch", "50",
    "--unfreeze-epoch", "150",
    "--freeze-batch-size", "8",
    "--unfreeze-batch-size", "4",
    "--freeze-train", "true",
    "--init-lr", "0.003",
    "--lr-decay-type", "cos",
    "--save-period", "10",
    "--eval-period", "10",
    "--num-workers", "0",
    "--save-dir", "$out\weights",
    "--log-dir", "$out\logs"
)

Set-Content -LiteralPath "$out\train_command.txt" -Value "$python $($cmd -join ' ')" -Encoding UTF8
$process = Start-Process -FilePath $python -ArgumentList $cmd -WorkingDirectory $root -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log" -PassThru -WindowStyle Hidden
Set-Content -LiteralPath "$out\train_pid.txt" -Value $process.Id -Encoding ASCII
Write-Host "Started CLCS Ours-A-V3 training. PID=$($process.Id)"
Write-Host "Output: $out"
