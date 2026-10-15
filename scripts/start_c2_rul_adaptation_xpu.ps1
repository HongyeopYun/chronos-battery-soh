$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$output = Join-Path $repo 'results\c2-battery\rul-context-nominal80-adaptation-xpu-v1'
$runtime = Join-Path $output '.runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
if (Test-Path (Join-Path $output 'complete.json')) { Write-Output 'Already complete'; exit 0 }
$launchFile = Join-Path $runtime 'launch.json'
if (Test-Path $launchFile) {
    $previous = Get-Content $launchFile -Raw | ConvertFrom-Json
    $existing = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.ProcessName -eq 'python') { Write-Output ('Already running, PID ' + $previous.pid); exit 0 }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $runtime ('run-' + $stamp + '.stdout.log')
$stderr = Join-Path $runtime ('run-' + $stamp + '.stderr.log')
$arguments = @('-u', '-X', 'utf8', ('"' + (Join-Path $repo 'scripts\run_c2_rul_adaptation_xpu.py') + '"'), '--out', ('"' + $output + '"'))
$process = Start-Process -FilePath (Join-Path $repo '.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[pscustomobject]@{pid=$process.Id; stdout=$stdout; stderr=$stderr; output=$output} | ConvertTo-Json | Set-Content (Join-Path $runtime 'launch.json') -Encoding utf8
Get-Content (Join-Path $runtime 'launch.json')
