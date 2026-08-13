# Lumen Link Windows/WSL network bootstrap.
# The default invocation is read-only. Pass -Apply from an elevated PowerShell
# only after reviewing the reported interface and addresses.
[CmdletBinding()]
param(
    [switch]$Apply,
    [ValidateSet("Auto", "Mirrored", "Nat")]
    [string]$Mode = "Auto",
    [string]$InterfaceAlias = "",
    [string]$Distro = "Ubuntu",
    [string]$ThreadripperAddress = "192.168.50.1",
    [string]$LumenAddress = "192.168.50.2",
    [int]$WorkerPort = 8765,
    [ValidatePattern('^~(/[A-Za-z0-9._-]+)+$')]
    [string]$WslProjectRoot = "~/lumenengine",
    [switch]$EnableSshBootstrap,
    [int]$SshBootstrapPort = 9022,
    [switch]$RefreshOnly
)

$ErrorActionPreference = "Stop"
$WorkerFirewallName = "Lumen Link Worker"
$SshFirewallName = "Lumen Link SSH Bootstrap"
$TaskName = "Lumen Link WSL Network Refresh"
$ProgramRoot = Join-Path $env:ProgramData "LumenLink"
$RefreshScript = Join-Path $ProgramRoot "refresh-portproxy.ps1"
$DashboardShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Lumen Link Dashboard.lnk"
$LegacyDashboardShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Lumen Link Dashboard.url"
$DashboardIcon = Join-Path $ProgramRoot "lumen-link.ico"
$StartupLog = Join-Path $ProgramRoot "startup.log"
$DashboardAddress = "127.0.0.1"
$WslStartupCommand = "cd $WslProjectRoot && ./scripts/lumen-link-wsl startup --apply"
$WslUpdateCommand = "cd $WslProjectRoot && ./scripts/lumen-link-wsl update-if-needed --apply"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-WslAddress {
    $raw = (& wsl.exe -d $Distro -- hostname -I 2>$null) -join " "
    $addresses = $raw -split "\s+" | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' }
    if (-not $addresses) {
        throw "Could not determine the $Distro WSL IPv4 address. Start Ubuntu once and retry."
    }
    return $addresses[0]
}

function Get-EffectiveMode {
    if ($Mode -ne "Auto") { return $Mode }
    $wslConfig = Join-Path $env:USERPROFILE ".wslconfig"
    if ((Test-Path $wslConfig) -and
        ((Get-Content $wslConfig -Raw) -match '(?im)^\s*networkingMode\s*=\s*mirrored\s*$')) {
        return "Mirrored"
    }
    return "Nat"
}

function Show-Status {
    $effectiveMode = Get-EffectiveMode
    Write-Host "Lumen Link - Windows/WSL status"
    Write-Host "  requested mode:       $Mode"
    Write-Host "  effective mode:       $effectiveMode"
    Write-Host "  WSL distribution:     $Distro"
    Write-Host "  direct-link address:  $ThreadripperAddress/24"
    Write-Host "  allowed Lumen client: $LumenAddress"
    Write-Host "  worker endpoint:      http://${ThreadripperAddress}:$WorkerPort"
    if ($InterfaceAlias) {
        $adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction SilentlyContinue
        Write-Host "  interface:            $InterfaceAlias ($($adapter.Status))"
        $address = Get-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $ThreadripperAddress -ErrorAction SilentlyContinue
        Write-Host "  static address:       $(if ($address) { 'ready' } else { 'missing' })"
    } else {
        Write-Host "  interface:            not selected (use -InterfaceAlias)"
    }
    $workerRule = Get-NetFirewallRule -DisplayName $WorkerFirewallName -ErrorAction SilentlyContinue
    Write-Host "  worker firewall:      $(if ($workerRule) { 'ready' } else { 'missing' })"
    Write-Host "  NAT forwarding:"
    netsh interface portproxy show v4tov4
    Write-Host "  WSL distributions:"
    wsl.exe --list --verbose
    Write-Host ""
    Write-Host "No changes were made. Run this script from elevated PowerShell with"
    Write-Host "-Apply -InterfaceAlias '<dedicated Ethernet name>' after reviewing the deployment guide."
}

function Set-RestrictedFirewallRule {
    param([string]$DisplayName, [int]$Port)
    Remove-NetFirewallRule -DisplayName $DisplayName -ErrorAction SilentlyContinue
    New-NetFirewallRule `
        -DisplayName $DisplayName `
        -Direction Inbound `
        -Action Allow `
        -Enabled True `
        -Profile Any `
        -Protocol TCP `
        -LocalAddress $ThreadripperAddress `
        -LocalPort $Port `
        -RemoteAddress $LumenAddress | Out-Null
}

function Set-PortProxy {
    param(
        [int]$ListenPort,
        [int]$ConnectPort,
        [string]$ConnectAddress,
        [string]$ListenAddress = $ThreadripperAddress
    )
    netsh interface portproxy delete v4tov4 `
        listenaddress=$ListenAddress listenport=$ListenPort 2>$null | Out-Null
    netsh interface portproxy add v4tov4 `
        listenaddress=$ListenAddress listenport=$ListenPort `
        connectaddress=$ConnectAddress connectport=$ConnectPort | Out-Null
}

function Install-NatRefreshTask {
    New-Item -ItemType Directory -Force -Path $ProgramRoot | Out-Null
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $sshLines = if ($EnableSshBootstrap) {
        @"
netsh interface portproxy delete v4tov4 listenaddress=$ThreadripperAddress listenport=$SshBootstrapPort 2>`$null | Out-Null
netsh interface portproxy add v4tov4 listenaddress=$ThreadripperAddress listenport=$SshBootstrapPort connectaddress=`$wslAddress connectport=22 | Out-Null
"@
    } else { "" }
    $content = @"
`$ErrorActionPreference = "Stop"
`$startupReady = `$false
`$nextStartupAttempt = Get-Date
`$nextSourceCheck = (Get-Date).AddMinutes(5)
while (`$true) {
  try {
    if (-not `$startupReady -and (Get-Date) -ge `$nextStartupAttempt) {
      "`$(Get-Date -Format o) checking Git, configuration and research deployment" | Out-File "$StartupLog" -Append -Encoding UTF8
      & wsl.exe -d "$Distro" -- bash -lc '$WslStartupCommand' 2>&1 | Out-File "$StartupLog" -Append -Encoding UTF8
      if (`$LASTEXITCODE -ne 0) { throw "WSL startup verification failed with exit code `$LASTEXITCODE" }
      `$startupReady = `$true
    }
    if (`$startupReady) {
      if ((Get-Date) -ge `$nextSourceCheck) {
        "`$(Get-Date -Format o) checking for a newer idle-safe Lumen revision" | Out-File "$StartupLog" -Append -Encoding UTF8
        & wsl.exe -d "$Distro" -- bash -lc '$WslUpdateCommand' 2>&1 | Out-File "$StartupLog" -Append -Encoding UTF8
        if (`$LASTEXITCODE -ne 0) { throw "WSL source check failed with exit code `$LASTEXITCODE" }
        `$nextSourceCheck = (Get-Date).AddMinutes(5)
      }
      & wsl.exe -d "$Distro" -- bash -lc 'systemctl --user start lumen-link-worker.service' | Out-Null
    }
    `$raw = (& wsl.exe -d "$Distro" -- hostname -I) -join " "
    `$wslAddress = ((`$raw -split "\s+") | Where-Object { `$_ -match '^\d+\.\d+\.\d+\.\d+`$' })[0]
    if (-not `$wslAddress) { throw "Could not resolve the WSL address" }
    netsh interface portproxy delete v4tov4 listenaddress=$ThreadripperAddress listenport=$WorkerPort 2>`$null | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=$ThreadripperAddress listenport=$WorkerPort connectaddress=`$wslAddress connectport=$WorkerPort | Out-Null
    netsh interface portproxy delete v4tov4 listenaddress=$DashboardAddress listenport=$WorkerPort 2>`$null | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=$DashboardAddress listenport=$WorkerPort connectaddress=`$wslAddress connectport=$WorkerPort | Out-Null
$sshLines
  } catch {
    "`$(Get-Date -Format o) `$_" | Out-File "$StartupLog" -Append -Encoding UTF8
    `$nextStartupAttempt = (Get-Date).AddMinutes(5)
  }
  Start-Sleep -Seconds 30
}
"@
    Set-Content -Path $RefreshScript -Value $content -Encoding UTF8
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RefreshScript`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -RunLevel Highest `
        -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
}

function Install-MirroredStartupTask {
    New-Item -ItemType Directory -Force -Path $ProgramRoot | Out-Null
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $content = @"
`$ErrorActionPreference = "Stop"
`$startupReady = `$false
`$nextStartupAttempt = Get-Date
`$nextSourceCheck = (Get-Date).AddMinutes(5)
while (`$true) {
  try {
    if (-not `$startupReady -and (Get-Date) -ge `$nextStartupAttempt) {
      "`$(Get-Date -Format o) checking Git, configuration and research deployment" | Out-File "$StartupLog" -Append -Encoding UTF8
      & wsl.exe -d "$Distro" -- bash -lc '$WslStartupCommand' 2>&1 | Out-File "$StartupLog" -Append -Encoding UTF8
      if (`$LASTEXITCODE -ne 0) { throw "WSL startup verification failed with exit code `$LASTEXITCODE" }
      `$startupReady = `$true
    }
    if (`$startupReady) {
      if ((Get-Date) -ge `$nextSourceCheck) {
        "`$(Get-Date -Format o) checking for a newer idle-safe Lumen revision" | Out-File "$StartupLog" -Append -Encoding UTF8
        & wsl.exe -d "$Distro" -- bash -lc '$WslUpdateCommand' 2>&1 | Out-File "$StartupLog" -Append -Encoding UTF8
        if (`$LASTEXITCODE -ne 0) { throw "WSL source check failed with exit code `$LASTEXITCODE" }
        `$nextSourceCheck = (Get-Date).AddMinutes(5)
      }
      & wsl.exe -d "$Distro" -- bash -lc 'systemctl --user start lumen-link-worker.service' | Out-Null
    }
  } catch {
    "`$(Get-Date -Format o) `$_" | Out-File "$StartupLog" -Append -Encoding UTF8
    `$nextStartupAttempt = (Get-Date).AddMinutes(5)
  }
  Start-Sleep -Seconds 30
}
"@
    Set-Content -Path $RefreshScript -Value $content -Encoding UTF8
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RefreshScript`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -RunLevel Highest `
        -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
}

function Install-DashboardShortcut {
    New-Item -ItemType Directory -Force -Path $ProgramRoot | Out-Null
    Add-Type -AssemblyName System.Drawing
    $bitmap = [System.Drawing.Bitmap]::new(64, 64)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.Clear([System.Drawing.Color]::FromArgb(11, 17, 20))
    $orbitPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(101, 216, 207), 3)
    $accentPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(79, 136, 173), 3)
    $coreBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(131, 239, 229))
    $graphics.DrawEllipse($orbitPen, 7, 20, 50, 24)
    $graphics.DrawEllipse($accentPen, 20, 7, 24, 50)
    $graphics.FillEllipse($coreBrush, 25, 25, 14, 14)
    $iconHandle = $bitmap.GetHicon()
    $icon = [System.Drawing.Icon]::FromHandle($iconHandle)
    $stream = [System.IO.File]::Open($DashboardIcon, [System.IO.FileMode]::Create)
    try { $icon.Save($stream) } finally { $stream.Dispose() }
    $icon.Dispose()
    $coreBrush.Dispose()
    $accentPen.Dispose()
    $orbitPen.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()

    Remove-Item -Path $LegacyDashboardShortcut -Force -ErrorAction SilentlyContinue
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($DashboardShortcut)
    $shortcut.TargetPath = Join-Path $env:SystemRoot "explorer.exe"
    $shortcut.Arguments = "http://${DashboardAddress}:$WorkerPort/dashboard"
    $shortcut.WorkingDirectory = $env:USERPROFILE
    $shortcut.IconLocation = "$DashboardIcon,0"
    $shortcut.Description = "Lumen Link local dashboard"
    $shortcut.Save()
}

if (-not $Apply) {
    Show-Status
    exit 0
}
if (-not (Test-IsAdministrator)) {
    throw "Open PowerShell with Run as administrator, then repeat this command."
}

$effectiveMode = Get-EffectiveMode
if ($RefreshOnly) {
    if ($effectiveMode -eq "Nat") {
        $wslAddress = Get-WslAddress
        Set-PortProxy -ListenPort $WorkerPort -ConnectPort $WorkerPort -ConnectAddress $wslAddress
        Set-PortProxy -ListenPort $WorkerPort -ConnectPort $WorkerPort -ConnectAddress $wslAddress -ListenAddress $DashboardAddress
        if ($EnableSshBootstrap) {
            Set-PortProxy -ListenPort $SshBootstrapPort -ConnectPort 22 -ConnectAddress $wslAddress
        }
        Install-NatRefreshTask
    } else {
        Install-MirroredStartupTask
    }
    Install-DashboardShortcut
    Write-Host "Lumen Link startup automation and local dashboard shortcut refreshed."
    Write-Host "  Dashboard: http://${DashboardAddress}:$WorkerPort/dashboard"
    Write-Host "  Startup:   Git check, configure, verify, then start"
    exit 0
}
if (-not $InterfaceAlias) {
    throw "-InterfaceAlias is required with -Apply. Use Get-NetAdapter to find the dedicated Ethernet port."
}
$adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction Stop
if ($adapter.Status -eq "Disabled") {
    Enable-NetAdapter -Name $InterfaceAlias -Confirm:$false
}
$foreignAddress = Get-NetIPAddress `
    -InterfaceAlias $InterfaceAlias `
    -AddressFamily IPv4 `
    -ErrorAction SilentlyContinue | Where-Object {
        $_.IPAddress -ne $ThreadripperAddress -and $_.IPAddress -notlike '169.254.*'
    }
if ($foreignAddress) {
    throw "The selected interface already has another IPv4 address. Confirm that this is the unused direct-link port before changing it."
}
$defaultRoute = Get-NetRoute `
    -InterfaceAlias $InterfaceAlias `
    -AddressFamily IPv4 `
    -DestinationPrefix "0.0.0.0/0" `
    -ErrorAction SilentlyContinue
if ($defaultRoute) {
    throw "The selected interface has a default route. Choose the unused Ethernet port; Lumen Link never changes an internet route."
}
Set-NetIPInterface `
    -InterfaceAlias $InterfaceAlias `
    -AddressFamily IPv4 `
    -Dhcp Disabled
Set-DnsClientServerAddress `
    -InterfaceAlias $InterfaceAlias `
    -ServerAddresses @()
if (-not (Get-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $ThreadripperAddress -ErrorAction SilentlyContinue)) {
    New-NetIPAddress `
        -InterfaceAlias $InterfaceAlias `
        -IPAddress $ThreadripperAddress `
        -PrefixLength 24 `
        -AddressFamily IPv4 | Out-Null
}
Set-RestrictedFirewallRule -DisplayName $WorkerFirewallName -Port $WorkerPort

if ($effectiveMode -eq "Nat") {
    $wslAddress = Get-WslAddress
    Set-PortProxy -ListenPort $WorkerPort -ConnectPort $WorkerPort -ConnectAddress $wslAddress
    Set-PortProxy -ListenPort $WorkerPort -ConnectPort $WorkerPort -ConnectAddress $wslAddress -ListenAddress $DashboardAddress
    if ($EnableSshBootstrap) {
        Set-PortProxy -ListenPort $SshBootstrapPort -ConnectPort 22 -ConnectAddress $wslAddress
        Set-RestrictedFirewallRule -DisplayName $SshFirewallName -Port $SshBootstrapPort
    }
    Install-NatRefreshTask
} else {
    netsh interface portproxy delete v4tov4 `
        listenaddress=$ThreadripperAddress listenport=$WorkerPort 2>$null | Out-Null
    netsh interface portproxy delete v4tov4 `
        listenaddress=$DashboardAddress listenport=$WorkerPort 2>$null | Out-Null
    $hyperVCommand = Get-Command New-NetFirewallHyperVRule -ErrorAction SilentlyContinue
    if ($hyperVCommand) {
        if (-not $hyperVCommand.Parameters.ContainsKey("RemoteAddresses")) {
            throw "This Windows build cannot restrict the mirrored WSL firewall by source. Use -Mode Nat."
        }
        Remove-NetFirewallHyperVRule -Name "LumenLinkWorker" -ErrorAction SilentlyContinue
        New-NetFirewallHyperVRule `
            -Name "LumenLinkWorker" `
            -DisplayName $WorkerFirewallName `
            -Direction Inbound `
            -Action Allow `
            -VMCreatorId "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}" `
            -Protocol TCP `
            -LocalPorts $WorkerPort `
            -RemoteAddresses $LumenAddress | Out-Null
    } else {
        throw "Mirrored WSL firewall controls are unavailable. Use -Mode Nat."
    }
    Install-MirroredStartupTask
}

Install-DashboardShortcut

Write-Host "Lumen Link Windows network applied."
Write-Host "  Mode:      $effectiveMode"
Write-Host "  Endpoint:  http://${ThreadripperAddress}:$WorkerPort"
Write-Host "  Allowed:   $LumenAddress only"
Write-Host "  Dashboard: http://${DashboardAddress}:$WorkerPort/dashboard"
Write-Host "  Watchdog:  repairs WSL worker and forwarding every 30 seconds"
Write-Host "No gateway or DNS was assigned to the direct Ethernet interface."
