[CmdletBinding()]
param(
    [switch]$VisibleSimulator,
    [switch]$Cpu,
    [switch]$NoDisplay,
    [ValidateRange(0, 65535)]
    [int]$WebPort = 0,
    [string]$Model,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$workspaceRoot = (Resolve-Path (Join-Path $projectRoot '..\..')).Path
$simulatorRoot = Join-Path $workspaceRoot '1.1.1'
$simulatorLauncher = Join-Path $simulatorRoot 'start-simulator.ps1'
$ipcDirectory = Join-Path $simulatorRoot 'runtime\talos-ipc'
$detector = Join-Path $projectRoot 'build\windows-vs2022\Release\yolo_detect.exe'
$defaultModel = Join-Path $workspaceRoot 'trains\models\legacy_robot_detection\0526.onnx'

if (-not (Test-Path -LiteralPath $simulatorLauncher)) {
    throw "Windows simulator launcher is missing: $simulatorLauncher"
}

function Test-LocalTcpPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $connection.AsyncWaitHandle.WaitOne(200)) { return $false }
        $client.EndConnect($connection)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Test-LocalUdpListener([int]$Port) {
    return @(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue).Count -gt 0
}

if (-not (Test-Path -LiteralPath $detector)) {
    if ($SkipBuild) {
        throw "yolo_detect.exe is missing and -SkipBuild was supplied: $detector"
    }
    & cmake --preset windows-vs2022 -S $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
    & cmake --build (Join-Path $projectRoot 'build\windows-vs2022') `
        --config Release
    if ($LASTEXITCODE -ne 0) { throw 'yoloDetect build failed.' }
}

$imageTransportReady = Test-LocalTcpPort 5602
$sceneControlReady = Test-LocalUdpListener 5603
if ($imageTransportReady -ne $sceneControlReady) {
    throw "Incomplete Daedalus simulator instance: TCP image port 5602=$imageTransportReady, scene-control UDP port 5603=$sceneControlReady. Stop the stale instance before restarting."
}

if (-not $imageTransportReady) {
    $simulatorArguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $simulatorLauncher,
        '-IpcDir', $ipcDirectory
    )
    if ($VisibleSimulator) { $simulatorArguments += '-Visible' }
    $simulator = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList $simulatorArguments -PassThru
    Write-Host "Started Daedalus simulator (PID $($simulator.Id))."

    $ready = $false
    $stableReadyChecks = 0
    for ($attempt = 0; $attempt -lt 100; ++$attempt) {
        if ((Test-LocalTcpPort 5602) -and (Test-LocalUdpListener 5603)) {
            ++$stableReadyChecks
            if ($stableReadyChecks -ge 5) {
                $ready = $true
                break
            }
        } else {
            $stableReadyChecks = 0
        }
        Start-Sleep -Milliseconds 200
    }
    if (-not $ready) {
        throw 'Daedalus simulator did not open both TCP image port 5602 and scene-control UDP port 5603 within 20 seconds.'
    }
} else {
    Write-Host 'Reusing the simulator listening on TCP 5602 and UDP 5603.'
}

$arguments = @('--ipc-dir', $ipcDirectory)
$arguments += @('--imgsz', '640')
if ($Cpu) { $arguments += '--cpu' }
if ($NoDisplay) { $arguments += '--no-display' }
if ($WebPort -ne 0) { $arguments += @('--web', $WebPort) }
if ([string]::IsNullOrWhiteSpace($Model)) { $Model = $defaultModel }
if (-not (Test-Path -LiteralPath $Model)) {
    throw "ONNX model is missing: $Model"
}
$arguments += @('--model', $Model)

Write-Host "Starting yoloDetect with IPC directory: $ipcDirectory"
& $detector @arguments
exit $LASTEXITCODE
