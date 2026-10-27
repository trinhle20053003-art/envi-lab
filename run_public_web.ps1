# run_public_web.ps1 - Khởi chạy EnVi Lab và mở URL Public Cloudflare
$Host.UI.RawUI.WindowTitle = "EnVi Lab - Public Web Server"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "        KHOI DONG ENVI LAB - PUBLIC WEB SERVER            " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

$env:KERAS_HOME = "$PSScriptRoot\backend\.keras"
$env:PYTHONIOENCODING = "utf-8"

# 1. Khởi động Backend FastAPI phục vụ cả API và Frontend React
Write-Host "[1/2] Dang nap model va khoi dong Backend..." -ForegroundColor Yellow
$backend = Start-Process -FilePath "$PSScriptRoot\backend\venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8008" `
    -PassThru

Start-Sleep -Seconds 8

# 2. Khởi động Cloudflare Tunnel cấp phát link HTTPS công khai
Write-Host "[2/2] Dang mo duong truyen HTTPS cong khai (Cloudflare Tunnel)..." -ForegroundColor Yellow
$logFile = "$PSScriptRoot\cloudflared.log"
if (Test-Path $logFile) { Remove-Item -Force $logFile }

$tunnel = Start-Process -FilePath "$PSScriptRoot\cloudflared.exe" `
    -ArgumentList "tunnel", "--url", "http://127.0.0.1:8008", "--logfile", $logFile `
    -PassThru

# Chờ và lấy link công khai
$publicUrl = ""
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $logFile) {
        $match = Select-String -Path $logFile -Pattern "https://[a-zA-Z0-9-]+\.trycloudflare\.com"
        if ($match) {
            $publicUrl = $match.Matches[0].Value
            break
        }
    }
}

if ($publicUrl) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  WEBSITE ENVI LAB DANG ONLINE CONG KHAI TAI:            " -ForegroundColor White
    Write-Host "  $publicUrl" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Bat ky ai tren the gioi (dien thoai, may tinh khac) deu co the truy cap link nay!" -ForegroundColor Green
    Write-Host "Nhan Enter hoac dong cua so nay khi muon dung server." -ForegroundColor Gray
    Start-Process $publicUrl
} else {
    Write-Host "Chua lay duoc link, vui long kiem tra file log: $logFile" -ForegroundColor Red
}

# Đợi người dùng tắt
try {
    Wait-Process -Id $backend.Id
} finally {
    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
}
