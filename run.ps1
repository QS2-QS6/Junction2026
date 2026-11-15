$sdk = $env:ROKOKO_SDK_HOME
& "$sdk\bin\rokoko.exe" activate

# Glove driver (drop --auto-stream-usb if the glove is on Wi-Fi)
$driver = Start-Process "$sdk\bin\rokoko-sdk.exe" -ArgumentList "--auto-stream-usb" -PassThru
Start-Sleep 2

# Visualizer/music server, run from the project folder
$server = Start-Process python -ArgumentList "glove_viz\server.py --no-browser" -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow
Start-Sleep 2
Start-Process "http://127.0.0.1:8000/"

Write-Host "Running. Press Enter to stop."
Read-Host | Out-Null
Stop-Process $server.Id, $driver.Id -ErrorAction SilentlyContinue
