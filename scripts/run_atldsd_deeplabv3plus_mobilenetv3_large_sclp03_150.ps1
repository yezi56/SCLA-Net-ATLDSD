$ErrorActionPreference = "Stop"
$root = "D:\Code\ATLDSD"
$env:PYTHONPATH = "$root\src;$root\src\models\deeplabv3plus;$root\src\modules"
$python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$out = "$root\outputs\atldsd\deeplabv3plus_mobilenetv3_large_sclp03_150"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$cmd = @(
    "$root\src\models\deeplabv3plus\train.py",
    "--cuda", "true",
    "--seed", "11",
    "--num-classes", "6",
    "--backbone", "mobilenetv3_large",
    "--pretrained", "true",
    "--downsample-factor", "16",
    "--attention-type", "none",
    "--use-ppm", "false",
    "--input-shape", "256", "256",
    "--init-epoch", "0",
    "--freeze-epoch", "50",
    "--freeze-batch-size", "8",
    "--unfreeze-epoch", "150",
    "--unfreeze-batch-size", "4",
    "--freeze-train", "true",
    "--init-lr", "0.003",
    "--optimizer-type", "sgd",
    "--lr-decay-type", "cos",
    "--save-period", "10",
    "--eval-period", "10",
    "--dataset-name", "ATLDSD",
    "--vocdevkit-path", "D:\dataset\ATLDSD\VOCdevkit",
    "--dice-loss", "true",
    "--focal-loss", "false",
    "--sclp", "true",
    "--sclp-prob", "0.3",
    "--sclp-max-components", "2",
    "--sclp-class-weights", "1.0", "2.0", "2.0", "3.0",
    "--num-workers", "0",
    "--auto-export-report", "true",
    "--report-dir", "$out\reports\best_val",
    "--report-split", "val",
    "--report-fps-interval", "100",
    "--save-dir", "$out\weights",
    "--log-dir", "$out\logs",
    "--class-names", "background", "leaf", "rust", "alternaria_leaf_spot", "gray_spot", "brown_spot"
)

Set-Content -LiteralPath "$out\train_command.txt" -Value "$python $($cmd -join ' ')" -Encoding UTF8
$process = Start-Process -FilePath $python -ArgumentList $cmd -WorkingDirectory $root -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log" -PassThru -WindowStyle Hidden
Set-Content -LiteralPath "$out\train_pid.txt" -Value $process.Id -Encoding ASCII
Write-Host "Started DeepLabV3+ MobileNetV3-Large SCLP03 training. PID=$($process.Id)"
Write-Host "Output: $out"
