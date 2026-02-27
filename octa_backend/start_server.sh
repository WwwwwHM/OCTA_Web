#!/bin/bash
# OCTA后端服务启动脚本（Linux/Mac）

echo "========================================"
echo "OCTA后端服务启动脚本"
echo "========================================"
echo ""

# 检查虚拟环境是否激活
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[警告] 未检测到虚拟环境"
    echo "正在尝试激活虚拟环境..."
    
    # 尝试激活虚拟环境
    if [ -f "../octa_env/bin/activate" ]; then
        source ../octa_env/bin/activate
        echo "[成功] 虚拟环境已激活"
    else
        echo "[错误] 找不到虚拟环境，请手动激活"
        echo "激活命令: source ../octa_env/bin/activate"
        exit 1
    fi
fi

# 检查依赖是否安装
echo ""
echo "[信息] 检查依赖包..."
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] FastAPI未安装，正在安装依赖..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
fi

# 创建必要的目录
echo ""
echo "[信息] 检查目录结构..."
mkdir -p uploads
mkdir -p results
mkdir -p models/weights

# 启动服务
echo ""
echo "========================================"
echo "正在启动后端服务..."
echo "服务地址: http://127.0.0.1:8000"
echo "API文档: http://127.0.0.1:8000/docs"
echo "按 Ctrl+C 停止服务"
echo "========================================"
echo ""

python main.py
