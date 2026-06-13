$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$ModelRoot = Join-Path $ProjectRoot "src\models\deeplabv3plus"
$RunRoot = Join-Path $ProjectRoot "outputs\atldsd\deeplabv3plus_mobilenet_150"
$Weights = Join-Path $RunRoot "weights"
$Logs = Join-Path $RunRoot "logs"
$Reports = Join-Path $RunRoot "reports\best_val"

New-Item -ItemType Directory -Force -Path $Weights, $Logs, $Reports | Out-Null

& $Python (Join-Path $ModelRoot "train.py") `
  --cuda true `
  --seed 11 `
  --num-classes 6 `
  --backbone mobilenet `
  --pretrained false `
  --model-path (Join-Path $ModelRoot "model_data\deeplab_mobilenetv2.pth") `
  --downsample-factor 16 `
  --attention-type none `
  --use-ppm false `
  --input-shape 256 256 `
  --init-epoch 0 `
  --freeze-epoch 50 `
  --freeze-batch-size 8 `
  --unfreeze-epoch 150 `
  --unfreeze-batch-size 4 `
  --freeze-train true `
  --init-lr 0.0035 `
  --optimizer-type sgd `
  --lr-decay-type cos `
  --save-period 10 `
  --eval-period 10 `
  --dataset-name ATLDSD `
  --vocdevkit-path "D:\dataset\ATLDSD\VOCdevkit" `
  --dice-loss true `
  --focal-loss false `
  --num-workers 0 `
  --auto-export-report true `
  --report-dir $Reports `
  --report-split val `
  --report-fps-interval 100 `
  --save-dir $Weights `
  --log-dir $Logs `
  --class-names background leaf rust alternaria_leaf_spot gray_spot brown_spot
