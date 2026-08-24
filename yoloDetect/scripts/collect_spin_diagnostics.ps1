param(
  [int]$Samples = 150,
  [int]$IntervalMs = 100,
  [string]$Output = "build\runtime-logs\spin-diagnostics.csv"
)

$ErrorActionPreference = 'Stop'
$rows = [System.Collections.Generic.List[object]]::new()
Remove-Item -LiteralPath $Output -Force -ErrorAction SilentlyContinue

for ($index = 0; $index -lt $Samples; $index++) {
  $status = Invoke-RestMethod 'http://127.0.0.1:8080/api/status' -TimeoutSec 5
  $measurement = @($status.tracker_measurements |
    Where-Object { $_.ekf_input }) | Select-Object -First 1
  $row = [pscustomobject]@{
    sample = $index
    sequence = $status.source_sequence
    tracker_state = $status.tracker_state
    omega_rad_s = $status.tracker_omega_rad_s
    theta_rad = $status.tracker_theta_rad
    observation_count = $status.tracker_observation_count
    reliable_yaw_count = $status.reliable_yaw_count
    slot = if ($measurement) { $measurement.associated_slot } else { $null }
    yaw_valid = if ($measurement) { $measurement.has_inward_yaw } else { $false }
    yaw_std_rad = if ($measurement) { $measurement.yaw_std_rad } else { $null }
    yaw_innovation_rad = if ($measurement) { $measurement.yaw_innovation_rad } else { $null }
    measured_yaw_rad = if ($measurement) { $measurement.inward_yaw_rad } else { $null }
    predicted_yaw_rad = if ($measurement) { $measurement.predicted_yaw_rad } else { $null }
    reprojection_rms_px = if ($measurement) { $measurement.reprojection_rms_px } else { $null }
    consecutive_misses = $status.tracker_consecutive_misses
  }
  $rows.Add($row)
  Start-Sleep -Milliseconds $IntervalMs
}

$rows | Export-Csv -LiteralPath $Output -NoTypeInformation -Encoding utf8
$observed = @($rows | Where-Object { $_.observation_count -gt 0 })
$invalidYaw = @($observed | Where-Object { -not $_.yaw_valid })
$empty = @($rows | Where-Object { $_.observation_count -eq 0 })
$omega = @($rows | ForEach-Object { [double]$_.omega_rad_s })
$slotSequence = [System.Collections.Generic.List[int]]::new()
foreach ($row in $rows) {
  if ($null -ne $row.slot -and ($slotSequence.Count -eq 0 -or
      $slotSequence[$slotSequence.Count - 1] -ne $row.slot)) {
    $slotSequence.Add([int]$row.slot)
  }
}
$direction = 0
$reversals = 0
for ($index = 1; $index -lt $slotSequence.Count; $index++) {
  $delta = ($slotSequence[$index] - $slotSequence[$index - 1] + 4) % 4
  $nextDirection = if ($delta -eq 1) { 1 } elseif ($delta -eq 3) { -1 } else { 0 }
  if ($nextDirection -ne 0) {
    if ($direction -ne 0 -and $direction -ne $nextDirection) { $reversals++ }
    $direction = $nextDirection
  }
}

[pscustomobject]@{
  output = $Output
  samples = $rows.Count
  observed_samples = $observed.Count
  empty_observation_samples = $empty.Count
  unreliable_yaw_samples = $invalidYaw.Count
  omega_min_rad_s = ($omega | Measure-Object -Minimum).Minimum
  omega_max_rad_s = ($omega | Measure-Object -Maximum).Maximum
  omega_below_2_5_rad_s = @($rows | Where-Object { $_.omega_rad_s -lt 2.5 }).Count
  slot_sequence = ($slotSequence -join ',')
  slot_direction_reversals = $reversals
} | Format-List
