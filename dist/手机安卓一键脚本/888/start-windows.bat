@echo off
echo 正在启动LLM代理服务 - Windows版...
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：Python未安装或未添加到PATH
    echo 请安装Python并确保添加到系统PATH
    pause
    exit /b 1
)

:: 设置Windows友好的配置
set HOST=127.0.0.1
set PORT=8080
set WEB_PORT=8081

echo 服务配置：
echo   API端口: %PORT%
echo   Web端口: %WEB_PORT%
echo   访问地址: http://localhost:%WEB_PORT%
echo.

:: 启动服务
python app.py

pause