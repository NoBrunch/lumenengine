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
$DashboardShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Lumen Link Dashboard.url"
$DashboardAddress = "127.0.0.1"

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
    $sshLines = if ($EnableSshBootstrap) {
        @"
netsh interface portproxy delete v4tov4 listenaddress=$ThreadripperAddress listenport=$SshBootstrapPort 2>`$null | Out-Null
netsh interface portproxy add v4tov4 listenaddress=$ThreadripperAddress listenport=$SshBootstrapPort connectaddress=`$wslAddress connectport=22 | Out-Null
"@
    } else { "" }
    $content = @"
`$ErrorActionPreference = "Stop"
while (`$true) {
  try {
    & wsl.exe -d "$Distro" -- bash -lc 'systemctl --user start lumen-link-worker.service' | Out-Null
    `$raw = (& wsl.exe -d "$Distro" -- hostname -I) -join " "
    `$wslAddress = ((`$raw -split "\s+") | Where-Object { `$_ -match '^\d+\.\d+\.\d+\.\d+`$' })[0]
    if (-not `$wslAddress) { throw "Could not resolve the WSL address" }
    netsh interface portproxy delete v4tov4 listenaddress=$ThreadripperAddress listenport=$WorkerPort 2>`$null | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=$ThreadripperAddress listenport=$WorkerPort connectaddress=`$wslAddress connectport=$WorkerPort | Out-Null
    netsh interface portproxy delete v4tov4 listenaddress=$DashboardAddress listenport=$WorkerPort 2>`$null | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=$DashboardAddress listenport=$WorkerPort connectaddress=`$wslAddress connectport=$WorkerPort | Out-Null
$sshLines
  } catch { }
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
}

function Install-MirroredStartupTask {
    New-Item -ItemType Directory -Force -Path $ProgramRoot | Out-Null
    $content = @"
`$ErrorActionPreference = "Stop"
while (`$true) {
  try { & wsl.exe -d "$Distro" -- bash -lc 'systemctl --user start lumen-link-worker.service' | Out-Null } catch { }
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
}

function Install-DashboardShortcut {
    $content = "[InternetShortcut]`r`nURL=http://${DashboardAddress}:$WorkerPort/dashboard`r`n"
    Set-Content -Path $DashboardShortcut -Value $content -Encoding ASCII
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
    if ($effectiveMode -ne "Nat") { exit 0 }
    $wslAddress = Get-WslAddress
    Set-PortProxy -ListenPort $WorkerPort -ConnectPort $WorkerPort -ConnectAddress $wslAddress
    Set-PortProxy -ListenPort $WorkerPort -ConnectPort $WorkerPort -ConnectAddress $wslAddress -ListenAddress $DashboardAddress
    if ($EnableSshBootstrap) {
        Set-PortProxy -ListenPort $SshBootstrapPort -ConnectPort 22 -ConnectAddress $wslAddress
    }
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
