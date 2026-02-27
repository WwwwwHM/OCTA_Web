@echo off
REM ========================================
REM OCTA API联调测试启动脚本（Windows）
REM ========================================

echo.
echo ======================================================================
echo                   OCTA API完整联调测试
echo ======================================================================
echo.

REM 检查后端服务是否运行
echo [INFO] 检查后端服务...
curl -s http://127.0.0.1:8000/ >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 后端服务未启动！
    echo.
    echo 请先启动后端服务:
    echo   方式1: 双击 start_server.bat
    echo   方式2: 运行 python main.py
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] 后端服务运行正常
echo.

REM 检查测试图像
if not exist "test_data\test_image.png" (
    echo [WARNING] 未找到测试图像: test_data\test_image.png
    echo.
    echo 请提供测试图像:
    echo   1. 创建 test_data 目录
    echo   2. 将OCTA测试图像复制为 test_data\test_image.png
    echo   3. 建议尺寸: 256x256, 格式: PNG
    echo.
    set /p continue="是否继续（可能会失败）? (Y/N): "
    if /i not "%continue%"=="Y" (
        exit /b 1
    )
)

echo.

REM 激活虚拟环境
if exist "..\octa_env\Scripts\activate.bat" (
    call ..\octa_env\Scripts\activate.bat
)

REM 运行测试
echo ======================================================================
echo                         开始执行联调测试
echo ======================================================================
echo.

python test_seg_api.py

if errorlevel 1 (
    echo.
    echo [ERROR] 测试执行失败！
    echo.
    echo 故障排查:
    echo   1. 查看上方错误信息
    echo   2. 检查后端日志: logs\octa_backend.log
    echo   3. 验证测试文件路径正确
    echo.
) else (
    echo.
    echo ======================================================================
    echo                         ✅ 测试完成
    echo ======================================================================
    echo.
    echo 查看测试结果:
    echo   - 分割掩码: test_results\api_result_mask.png
    echo   - 对比图: test_results\comparison.png
    echo.
)

pause
