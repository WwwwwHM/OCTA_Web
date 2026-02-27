@echo off
REM OCTA后端服务启动脚本（Windows）

echo ========================================
echo OCTA后端服务启动脚本
echo ========================================
echo.

REM 检查虚拟环境是否激活
if "%VIRTUAL_ENV%"=="" (
    echo [警告] 未检测到虚拟环境
    echo 正在尝试激活虚拟环境...
    
    REM 尝试激活虚拟环境
    if exist "..\octa_env\Scripts\activate.bat" (
        call ..\octa_env\Scripts\activate.bat
        echo [成功] 虚拟环境已激活
    ) else (
        echo [错误] 找不到虚拟环境，请手动激活
        echo 激活命令: ..\octa_env\Scripts\activate
        pause
        exit /b 1
    )
)

REM 检查依赖是否安装
echo.
echo [信息] 检查依赖包...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [警告] FastAPI未安装，正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

REM 创建必要的目录
echo.
echo [信息] 检查目录结构...
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "models\weights" mkdir models\weights

REM 启动服务
echo.
echo ========================================
echo 正在启动后端服务...
echo 服务地址: http://127.0.0.1:8000
echo API文档: http://127.0.0.1:8000/docs
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

python main.py

pause
