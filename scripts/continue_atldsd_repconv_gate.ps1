param(
    [string]$Python = "D:\soft\Anaconda\envs\Pytorch\python.exe",
    [switch]$StartLowCa
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Reference11 = Join-Path $ProjectRoot "outputs\atldsd_fast\long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s11"
$Reference23 = Join-Path $ProjectRoot "outputs\atldsd_fast\long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s23"
$Candidate11 = Join-Path $ProjectRoot "outputs\atldsd_fast\long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s11"
$Candidate23 = Join-Path $ProjectRoot "outputs\atldsd_fast\long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s23"
$SummaryScript = Join-Path $ProjectRoot "scripts\summarize_atldsd_candidate.py"
$SummaryOut = Join-Path $ProjectRoot "outputs\atldsd_fast\repconv_vs_reference_current.md"
$CandidateReport = Join-Path $Candidate23 "reports\best_miou\metrics_summary.json"
$LowCaScript = Join-Path $ProjectRoot "scripts\run_fast_lgc_lcsf_sp384_balanced_low_ca_screen.ps1"
$RelayLog = Join-Path $ProjectRoot "seg\ATLDSD_training_relay_2026-06-10.md"

function Add-RelayLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm zzz"
    Add-Content -LiteralPath $RelayLog -Encoding UTF8 -Value "`n## $timestamp RepConv gate`n`n$Message"
}

& $Python $SummaryScript `
    --reference $Reference11 $Reference23 `
    --candidate $Candidate11 $Candidate23 `
    --reference-name "LGC-LCSF-SP384-BalancedPrefix-full-e80" `
    --candidate-name "RepConv-decoder-full-e80" `
    --output-md $SummaryOut | Write-Output

if (-not (Test-Path $CandidateReport)) {
    $bestMiouPath = Join-Path $Candidate23 "weights\best_miou.txt"
    $bestMiou = if (Test-Path $bestMiouPath) { Get-Content -LiteralPath $bestMiouPath -ErrorAction SilentlyContinue } else { "missing" }
    Add-RelayLog "RepConv seed23 report is not ready. Current best_miou.txt = $bestMiou. Decision remains WAIT; Low-CA is not started."
    exit 0
}

$summaryText = Get-Content -LiteralPath $SummaryOut -Raw -Encoding UTF8
if ($summaryText -match "Decision: UPGRADE") {
    Add-RelayLog "RepConv final report is ready and the summary gate says UPGRADE. Keep RepConv as the new candidate; Low-CA is not started."
    exit 0
}

if ($summaryText -notmatch "Decision: REJECT") {
    Add-RelayLog "RepConv final report is ready, but the summary gate did not produce a final REJECT/UPGRADE decision. Manual inspection required; Low-CA is not started."
    exit 0
}

if (-not $StartLowCa) {
    Add-RelayLog "RepConv final report is ready and the summary gate says REJECT. Low-CA was not started because -StartLowCa was not provided."
    exit 0
}

$repconvProcess = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s23*" }

if ($repconvProcess) {
    Add-RelayLog "RepConv summary gate says REJECT, but a RepConv seed23 python process is still present. Low-CA is not started."
    exit 0
}

powershell -ExecutionPolicy Bypass -File $LowCaScript -Python $Python
Add-RelayLog "RepConv summary gate says REJECT, RepConv process is gone, and Low-CA quick screen was started via `scripts/run_fast_lgc_lcsf_sp384_balanced_low_ca_screen.ps1`."
