[CmdletBinding()]
param(
    [switch]$NoDisplay,
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$workspaceRoot = (Resolve-Path (Join-Path $projectRoot '..\..')).Path
$simulatorRoot = Join-Path $workspaceRoot '1.1.1'
$ipcDirectory = Join-Path $simulatorRoot 'runtime\talos-ipc'
$simulatorLauncher = Join-Path $simulatorRoot 'start-simulator.ps1'
$detector = Join-Path $projectRoot 'build\windows-vs2022\Release\yolo_detect.exe'

function Test-TcpPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync('127.0.0.1', $Port)
        if (-not $task.Wait(300)) { return $false }
        return $client.Connected
    } catch { return $false }
    finally { $client.Dispose() }
}

function Test-UdpPort([int]$Port) {
    return @(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue).Count -gt 0
}

if (-not (Test-Path -LiteralPath $detector)) {
    throw "yolo_detect.exe not found: $detector"
}
if (-not (Test-Path -LiteralPath $simulatorLauncher)) {
    throw "Simulator launcher not found: $simulatorLauncher"
}

if (-not (Test-TcpPort 5602)) {
    $simulatorArgs = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $simulatorLauncher,
        '-IpcDir', $ipcDirectory, '-Visible'
    )
    Start-Process -FilePath 'powershell.exe' -ArgumentList $simulatorArgs | Out-Null
    Write-Host 'Starting DaedalusSimulator...'
}

$ready = $false
for ($attempt = 0; $attempt -lt 100; $attempt++) {
    if ((Test-TcpPort 5602) -and (Test-UdpPort 5603) -and
        (Test-Path -LiteralPath (Join-Path $ipcDirectory 'talos_ipc_meta'))) {
        $ready = $true
        break
    }
    Start-Sleep -Milliseconds 200
}
if (-not $ready) {
    throw 'DaedalusSimulator did not become ready within 20 seconds.'
}

if (Get-Process -Name yolo_detect -ErrorAction SilentlyContinue) {
    Write-Host 'yolo_detect is already running; reusing the existing web service.'
} else {
    $detectorArgs = @('--web', '8080', '--no-display', '--ipc-dir', $ipcDirectory)
    if ($Cpu) { $detectorArgs += '--cpu' }
    Start-Process -FilePath $detector -ArgumentList $detectorArgs -WorkingDirectory $projectRoot | Out-Null
    Write-Host 'Starting yolo_detect GPU web service...'
}

for ($attempt = 0; $attempt -lt 50; $attempt++) {
    if (Test-TcpPort 8080) { break }
    Start-Sleep -Milliseconds 200
}
Start-Process 'http://127.0.0.1:8080/'
Write-Host 'Web UI: http://127.0.0.1:8080/'
