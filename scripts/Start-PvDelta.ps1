[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateSet('127.0.0.1', '0.0.0.0')][string]$BindAddress = '0.0.0.0',
    [string]$SdkLibrary = '',
    [string]$SurfaceManifest = '',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project .venv is missing. Follow docs/stage1-pilot.md to install the dashboard."
}
if (-not $SdkLibrary) {
    $SdkLibrary = $env:PVDELTA_DIRP_LIB
}
if (-not $SdkLibrary) {
    $SdkLibrary = Join-Path $projectRoot 'sdk/tsdk-core/lib/windows/release_x64'
}
if (-not (Test-Path -LiteralPath $SdkLibrary)) {
    throw 'DJI SDK location is missing. Pass -SdkLibrary with the SDK library folder or DLL.'
}
$SdkLibrary = (Resolve-Path -LiteralPath $SdkLibrary).Path
& $pythonPath -c 'import sys; import streamlit, duckdb, folium, cv2, PIL; from pvdelta.decode import load_library; load_library(sys.argv[1]); print("Dashboard dependencies and DJI SDK: ready")' $SdkLibrary
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency/SDK check failed. Follow docs/stage1-pilot.md.'
}
if (-not $SurfaceManifest) { $SurfaceManifest = $env:PVDELTA_SURFACE_MANIFEST }
if (-not $SurfaceManifest) {
    $localSurfaceManifest = Join-Path $projectRoot 'data/surface-recovery/manifest.json'
    $container = Split-Path -Parent $projectRoot
    if ((Split-Path -Leaf $container) -eq 'GitHub') {
        $legacyWorkspace = Join-Path (Split-Path -Parent $container) 'knowledge basis'
        if (Test-Path -LiteralPath $legacyWorkspace -PathType Container) {
            $localSurfaceManifest = Join-Path $legacyWorkspace 'surface-recovery/manifest.json'
        }
    }
    if (Test-Path -LiteralPath $localSurfaceManifest -PathType Leaf) { $SurfaceManifest = $localSurfaceManifest }
}
if ($SurfaceManifest) {
    $SurfaceManifest = (Resolve-Path -LiteralPath $SurfaceManifest).Path
    & $pythonPath -c 'import sys; from pvdelta.surface_constraints import configured_surface_index; digest, refs = configured_surface_index(sys.argv[1]); print("Source-bound surface references ready:", len(refs), "frames")' $SurfaceManifest
    if ($LASTEXITCODE -ne 0) { throw 'Surface reference configuration is invalid.' }
}
if ($CheckOnly) { return }
$env:PVDELTA_DIRP_LIB = $SdkLibrary
if ($SurfaceManifest) { $env:PVDELTA_SURFACE_MANIFEST = $SurfaceManifest }
$appPath = Join-Path $projectRoot 'dashboard/flight_planner.py'
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m streamlit run $appPath --server.address $BindAddress --server.port $Port --server.headless true
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit could not start; check whether the port is already in use.' }
}
finally { Pop-Location }
