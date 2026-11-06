@echo off
echo ========================================================
echo Starting Public Cloudflare Tunnel for FastAPI Application
echo Local Port: 8001 -> Public Worldwide HTTPS URL
echo ========================================================
echo.
echo Make sure your Docker container or FastAPI app is running on port 8001.
echo Connecting to Cloudflare Edge...
echo.
cloudflared.exe tunnel --url http://localhost:8001
pause
