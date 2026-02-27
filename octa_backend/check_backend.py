"""
OCTA后端验证脚本

此脚本用于验证后端环境配置是否正确，包括：
- 虚拟环境检查
- 依赖包检查
- API接口测试

使用方法：
    python check_backend.py

作者：OCTA Web项目组
日期：2024
"""

import sys
import os
from pathlib import Path
import subprocess

# 颜色输出（Windows和Linux兼容）
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    # 如果没有colorama，使用简单的输出
    class Fore:
        GREEN = ''
        RED = ''
        YELLOW = ''
        BLUE = ''
        CYAN = ''
    class Style:
        RESET_ALL = ''
    HAS_COLORAMA = False


def print_success(message):
    """打印成功消息（绿色）"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message):
    """打印错误消息（红色）"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_warning(message):
    """打印警告消息（黄色）"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def print_info(message):
    """打印信息消息（蓝色）"""
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")


def check_virtual_env():
    """
    检查虚拟环境是否激活
    
    Returns:
        bool: 如果虚拟环境已激活返回True，否则返回False
    """
    print("\n" + "=" * 60)
    print("步骤1: 检查虚拟环境")
    print("=" * 60)
    
    # 检查VIRTUAL_ENV环境变量
    if os.environ.get('VIRTUAL_ENV'):
        venv_path = os.environ.get('VIRTUAL_ENV')
        print_success(f"虚拟环境已激活: {venv_path}")
        return True
    else:
        # 检查是否在conda环境
        if os.environ.get('CONDA_DEFAULT_ENV'):
            conda_env = os.environ.get('CONDA_DEFAULT_ENV')
            print_success(f"Conda环境已激活: {conda_env}")
            return True
        else:
            # 检查Python路径是否包含venv或env
            python_path = sys.executable
            if 'venv' in python_path.lower() or 'env' in python_path.lower():
                print_success(f"检测到虚拟环境路径: {python_path}")
                return True
            else:
                print_error("未检测到虚拟环境")
                print_warning("建议：激活虚拟环境后再运行此脚本")
                print_info("Windows激活命令: .\\octa_env\\Scripts\\activate")
                print_info("Linux/Mac激活命令: source octa_env/bin/activate")
                return False


def check_dependencies():
    """
    检查必要的依赖包是否已安装
    
    Returns:
        bool: 如果所有依赖都已安装返回True，否则返回False
    """
    print("\n" + "=" * 60)
    print("步骤2: 检查依赖包")
    print("=" * 60)
    
    # 注意：包名和导入名可能不同
    # Pillow安装后导入时使用PIL，不是pillow
    required_packages = {
        'fastapi': ('fastapi', 'FastAPI'),
        'uvicorn': ('uvicorn', 'Uvicorn'),
        'pillow': ('PIL', 'Pillow (PIL)'),  # 安装名是pillow，导入名是PIL
        'numpy': ('numpy', 'NumPy'),
        'torch': ('torch', 'PyTorch'),
        'torchvision': ('torchvision', 'TorchVision'),
        'requests': ('requests', 'Requests (用于API测试)'),
    }
    
    missing_packages = []
    installed_packages = []
    
    for package_key, (import_name, display_name) in required_packages.items():
        try:
            __import__(import_name)
            print_success(f"{display_name} 已安装")
            installed_packages.append(package_key)
        except ImportError:
            print_error(f"{display_name} 未安装")
            missing_packages.append(package_key)
    
    if missing_packages:
        print_warning(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print_info("安装命令: pip install -r requirements.txt")
        return False
    else:
        print_success("\n所有依赖包已安装")
        return True


def check_directories():
    """
    检查必要的目录是否存在
    
    Returns:
        bool: 如果所有目录都存在返回True，否则返回False
    """
    print("\n" + "=" * 60)
    print("步骤3: 检查目录结构")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    required_dirs = {
        'models': '模型目录',
        'models/weights': '模型权重目录',
        'uploads': '上传文件目录',
        'results': '结果文件目录',
    }
    
    all_exist = True
    
    for dir_path, description in required_dirs.items():
        full_path = base_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            print_success(f"{description}: {full_path}")
        else:
            print_error(f"{description}不存在: {full_path}")
            # 尝试创建目录
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print_info(f"已自动创建: {full_path}")
            except Exception as e:
                print_error(f"创建失败: {e}")
                all_exist = False
    
    return all_exist


def test_api_upload():
    """
    测试API接口的文件上传功能
    
    Returns:
        bool: 如果测试成功返回True，否则返回False
    """
    print("\n" + "=" * 60)
    print("步骤4: 测试API接口")
    print("=" * 60)
    
    # 检查后端服务是否运行
    import requests
    
    base_url = "http://127.0.0.1:8000"
    
    # 测试健康检查接口
    print_info("测试健康检查接口: GET /")
    print_info(f"尝试连接: {base_url}")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print_success("健康检查接口正常")
            try:
                print_info(f"响应: {response.json()}")
            except:
                print_info(f"响应: {response.text[:100]}...")
        else:
            print_error(f"健康检查失败，状态码: {response.status_code}")
            print_error(f"响应内容: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("无法连接到后端服务")
        print_warning("后端服务未启动或无法访问")
        print_info("启动方法1: 在另一个终端运行 'python main.py'")
        print_info("启动方法2: 使用启动脚本 'start_server.bat' (Windows) 或 './start_server.sh' (Linux/Mac)")
        print_info("启动方法3: 使用uvicorn 'uvicorn main:app --host 127.0.0.1 --port 8000 --reload'")
        print_warning("注意: 请先启动后端服务，然后再运行此验证脚本")
        return False
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False
    
    # 测试文件上传接口（创建模拟PNG文件）
    print_info("\n测试文件上传接口: POST /segment-octa/")
    
    # 创建一个简单的测试PNG图像
    try:
        from PIL import Image
        import numpy as np
        
        # 创建256x256的测试图像
        test_image = Image.new('RGB', (256, 256), color='white')
        test_image_path = Path(__file__).parent / "test_image.png"
        test_image.save(test_image_path)
        print_success(f"创建测试图像: {test_image_path}")
        
        # 准备上传文件
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            data = {'model_type': 'unet'}
            
            print_info("发送上传请求...")
            response = requests.post(
                f"{base_url}/segment-octa/",
                files=files,
                data=data,
                timeout=60  # 图像处理可能需要较长时间
            )
        
        # 清理测试文件
        if test_image_path.exists():
            test_image_path.unlink()
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print_success("文件上传和分割接口测试成功")
                print_info(f"结果URL: {result.get('result_url')}")
            else:
                print_warning("接口返回成功，但分割可能失败（模型未训练）")
                print_info(f"消息: {result.get('message')}")
            return True
        else:
            print_error(f"上传失败，状态码: {response.status_code}")
            print_error(f"响应: {response.text}")
            return False
            
    except ImportError:
        print_error("无法导入PIL或numpy，跳过文件上传测试")
        return False
    except Exception as e:
        print_error(f"文件上传测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：运行所有检查"""
    print("\n" + "=" * 60)
    print("OCTA后端环境验证脚本")
    print("=" * 60)
    
    results = {
        '虚拟环境': check_virtual_env(),
        '依赖包': check_dependencies(),
        '目录结构': check_directories(),
        'API接口': test_api_upload(),
    }
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: 通过")
        else:
            print_error(f"{check_name}: 失败")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print_success("所有检查通过！后端环境配置正确。")
    else:
        print_warning("部分检查未通过，请根据上述提示修复问题。")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n验证脚本执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
