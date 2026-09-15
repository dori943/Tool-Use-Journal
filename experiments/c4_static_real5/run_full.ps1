param([Parameter(Mandatory=$true)][string]$PrepDir)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Prep = (Resolve-Path $PrepDir).Path
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$Manifest = Join-Path $Prep 'selected_static_inputs.csv'
$Config = Join-Path $Prep 'inference_config.yaml'
$Folder = Join-Path $Prep 'full_inference'
New-Item -ItemType Directory -Path $Folder -Force | Out-Null
$Files = @()
for ($i = 0; $i -lt 5; $i++) {
    $Output = Join-Path $Folder "repeat_$i.jsonl"
    if (Test-Path $Output) { throw "Existing prediction file; inspect rather than overwrite: $Output" }
    & $Python (Join-Path $PSScriptRoot 'infer.py') --config $Config --manifest $Manifest --output $Output --repeat-id $i
    if ($LASTEXITCODE -ne 0) { throw "Inference repeat $i failed" }
    $Files += $Output
}
$Result = Join-Path $Folder 'evaluation.json'
if (Test-Path $Result) { throw "Existing evaluation; inspect rather than overwrite: $Result" }
& $Python (Join-Path $PSScriptRoot 'evaluate_static.py') --prep-dir $Prep --predictions $Files[0] $Files[1] $Files[2] $Files[3] $Files[4] --output $Result
if ($LASTEXITCODE -ne 0) { throw 'Static Real-5 evaluation failed' }
