"""
OCTA血管分割API完整联调测试脚本

功能：
1. 测试权重上传接口 (/api/v1/weight/upload)
2. 测试分割预测接口 (/api/v1/seg/predict)
3. 解析mask_base64并保存为本地图片
4. 对比与本地模型推理结果的一致性

使用方法：
    # 启动后端服务（终端1）
    cd octa_backend
    python main.py
    
    # 运行测试脚本（终端2）
    cd octa_backend
    python test_seg_api.py

作者：OCTA Web项目组
日期：2026-01-28
"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ==================== 测试配置 ====================

# 后端服务地址
BASE_URL = "http://127.0.0.1:8000"

# 测试文件路径（请根据实际情况修改）
TEST_WEIGHT_PATH = "./models/weights/unet_octa.pth"  # 测试权重文件
TEST_IMAGE_PATH = "./test_data/test_image.png"       # 测试OCTA图像

# 输出目录
OUTPUT_DIR = Path("./test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 辅助函数 ====================

def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}".center(80))
    print("=" * 80)


def print_step(step_num: int, description: str):
    """打印步骤信息"""
    print(f"\n[步骤 {step_num}] {description}")
    print("-" * 80)


def check_server_health() -> bool:
    """检查后端服务是否正常运行"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 服务状态: {data.get('status')}")
            print(f"✓ 服务信息: {data.get('message')}")
            return True
        else:
            print(f"✗ 服务返回异常状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到后端服务")
        print("  请确保后端服务已启动：cd octa_backend && python main.py")
        return False
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False


def upload_weight(weight_path: str) -> str:
    """
    上传权重文件到后端
    
    Args:
        weight_path: 本地权重文件路径
        
    Returns:
        weight_id: 上传成功后返回的权重ID
        
    Raises:
        Exception: 上传失败时抛出异常
    """
    print(f"📤 正在上传权重文件: {weight_path}")
    
    # 检查文件是否存在
    if not Path(weight_path).exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    # 构造multipart/form-data请求
    with open(weight_path, "rb") as f:
        files = {
            "file": (Path(weight_path).name, f, "application/octet-stream")
        }
        data = {
            "description": "联调测试上传的权重文件"
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/weight/upload",
                files=files,
                data=data,
                timeout=60  # 权重文件可能较大，设置60秒超时
            )
            
            if response.status_code == 200:
                result = response.json()
                weight_id = result.get("data", {}).get("weight_id")
                print(f"✓ 权重上传成功")
                print(f"  Weight ID: {weight_id}")
                print(f"  文件大小: {result.get('data', {}).get('size')} bytes")
                return weight_id
            else:
                print(f"✗ 权重上传失败: {response.status_code}")
                print(f"  响应内容: {response.text}")
                raise Exception(f"上传失败: {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("请求超时（>60秒），权重文件可能过大")
        except requests.exceptions.RequestException as e:
            raise Exception(f"网络请求失败: {e}")


def predict_segmentation(image_path: str, weight_id: str = None) -> dict:
    """
    调用分割预测接口
    
    Args:
        image_path: 本地测试图像路径
        weight_id: 权重ID，None表示使用官方权重
        
    Returns:
        dict: 预测结果，包含mask_base64等字段
        
    Raises:
        Exception: 预测失败时抛出异常
    """
    print(f"🔍 正在调用分割预测接口")
    print(f"  图像路径: {image_path}")
    print(f"  权重ID: {weight_id or 'official（官方权重）'}")
    
    # 检查图像文件是否存在
    if not Path(image_path).exists():
        raise FileNotFoundError(f"测试图像不存在: {image_path}")
    
    # 构造multipart/form-data请求
    with open(image_path, "rb") as f:
        files = {
            "image_file": (Path(image_path).name, f, "image/png")
        }
        data = {}
        if weight_id:
            data["weight_id"] = weight_id
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/api/v1/seg/predict",
                files=files,
                data=data,
                timeout=120  # 推理可能耗时，设置120秒超时
            )
            
            # 记录结束时间
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 预测成功")
                print(f"  总耗时: {elapsed_time:.2f}秒（含网络传输）")
                print(f"  服务器推理耗时: {result.get('data', {}).get('infer_time')}秒")
                print(f"  推理设备: {result.get('data', {}).get('device')}")
                print(f"  使用权重: {result.get('data', {}).get('weight_id')}")
                return result.get("data", {})
            else:
                print(f"✗ 预测失败: {response.status_code}")
                print(f"  响应内容: {response.text}")
                raise Exception(f"预测失败: {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("请求超时（>120秒），模型推理可能卡住")
        except requests.exceptions.RequestException as e:
            raise Exception(f"网络请求失败: {e}")


def decode_and_save_mask(mask_base64: str, output_path: str) -> np.ndarray:
    """
    解码Base64掩码并保存为图片
    
    Args:
        mask_base64: Base64编码的掩码字符串
        output_path: 输出图片保存路径
        
    Returns:
        np.ndarray: 解码后的掩码数组
    """
    print(f"💾 正在解码并保存掩码")
    print(f"  Base64长度: {len(mask_base64)} 字符")
    print(f"  输出路径: {output_path}")
    
    try:
        # 解码Base64
        mask_bytes = base64.b64decode(mask_base64)
        
        # 从bytes创建PIL Image
        mask_image = Image.open(BytesIO(mask_bytes))
        
        # 转为numpy数组
        mask_array = np.array(mask_image)
        
        # 保存为PNG
        mask_image.save(output_path)
        
        print(f"✓ 掩码保存成功")
        print(f"  图像尺寸: {mask_image.size}")
        print(f"  数组形状: {mask_array.shape}")
        print(f"  数据类型: {mask_array.dtype}")
        print(f"  值范围: [{mask_array.min()}, {mask_array.max()}]")
        
        return mask_array
        
    except Exception as e:
        print(f"✗ 掩码解码失败: {e}")
        raise


def compare_with_local_inference(api_mask: np.ndarray, test_image_path: str):
    """
    对比API结果与本地推理结果（可选）
    
    Args:
        api_mask: API返回的掩码数组
        test_image_path: 测试图像路径
    
    Note:
        此函数需要本地模型和预处理代码，如不需要可跳过
    """
    print(f"⚖️  对比API结果与本地推理（可选）")
    
    try:
        # 导入本地模型和处理函数
        from models.unet import UNetUnderfittingFix, segment_octa_image
        
        # 本地推理
        print("  正在执行本地推理...")
        local_result_path = segment_octa_image(
            image_path=test_image_path,
            model_type="unet",
            output_path=str(OUTPUT_DIR / "local_result.png")
        )
        
        # 加载本地结果
        local_mask = np.array(Image.open(local_result_path))
        
        # 对比两个掩码
        diff = np.abs(api_mask.astype(np.int16) - local_mask.astype(np.int16))
        diff_ratio = (diff > 0).sum() / diff.size * 100
        
        print(f"✓ 本地推理完成")
        print(f"  差异像素比例: {diff_ratio:.2f}%")
        
        if diff_ratio < 0.1:
            print(f"  ✓ API结果与本地完全一致（差异<0.1%）")
        elif diff_ratio < 5:
            print(f"  ⚠ API结果与本地基本一致（差异<5%）")
        else:
            print(f"  ✗ API结果与本地存在较大差异（差异>{diff_ratio:.1f}%）")
            # 保存差异图
            diff_image = Image.fromarray((diff).astype(np.uint8))
            diff_path = OUTPUT_DIR / "diff_mask.png"
            diff_image.save(diff_path)
            print(f"  差异图已保存: {diff_path}")
        
    except ImportError:
        print("  ⚠ 本地模型未导入，跳过对比（非必需）")
    except Exception as e:
        print(f"  ⚠ 本地推理失败: {e}（非关键错误，继续）")


def visualize_result(original_path: str, mask_path: str):
    """
    可视化对比：原图 vs 分割结果
    
    Args:
        original_path: 原始图像路径
        mask_path: 掩码图像路径
    """
    print(f"📊 生成可视化对比图")
    
    try:
        # 加载图像
        original = Image.open(original_path).convert("L")
        mask = Image.open(mask_path)
        
        # 调整尺寸一致
        if original.size != mask.size:
            print(f"  原图尺寸: {original.size}")
            print(f"  掩码尺寸: {mask.size}")
            mask = mask.resize(original.size, Image.NEAREST)
        
        # 创建对比图（并排显示）
        width, height = original.size
        comparison = Image.new("L", (width * 2, height))
        comparison.paste(original, (0, 0))
        comparison.paste(mask, (width, 0))
        
        # 保存对比图
        comparison_path = OUTPUT_DIR / "comparison.png"
        comparison.save(comparison_path)
        
        print(f"✓ 可视化对比图已保存: {comparison_path}")
        
    except Exception as e:
        print(f"⚠ 可视化失败: {e}（非关键错误）")


# ==================== 主测试流程 ====================

def main():
    """主测试流程"""
    
    print_section("OCTA血管分割API完整联调测试")
    
    print("\n📋 测试环境信息")
    print(f"  后端地址: {BASE_URL}")
    print(f"  测试权重: {TEST_WEIGHT_PATH}")
    print(f"  测试图像: {TEST_IMAGE_PATH}")
    print(f"  输出目录: {OUTPUT_DIR}")
    
    # ==================== 步骤0：健康检查 ====================
    print_step(0, "后端服务健康检查")
    if not check_server_health():
        print("\n❌ 测试终止：后端服务未启动")
        print("\n启动方法：")
        print("  cd octa_backend")
        print("  python main.py")
        return
    
    try:
        # ==================== 步骤1：上传权重（可选）====================
        print_step(1, "上传权重文件（可选，也可使用官方权重）")
        
        # 选择权重方式
        use_uploaded_weight = False  # 设为True测试上传权重，False使用官方权重
        weight_id = None
        
        if use_uploaded_weight:
            if Path(TEST_WEIGHT_PATH).exists():
                try:
                    weight_id = upload_weight(TEST_WEIGHT_PATH)
                    print(f"✓ 将使用上传的权重: {weight_id}")
                except Exception as e:
                    print(f"⚠ 权重上传失败: {e}")
                    print(f"  改用官方权重进行测试")
                    weight_id = None
            else:
                print(f"⚠ 测试权重文件不存在: {TEST_WEIGHT_PATH}")
                print(f"  改用官方权重进行测试")
        else:
            print("ℹ️  跳过权重上传，使用官方预置权重")
        
        # ==================== 步骤2：调用分割预测接口 ====================
        print_step(2, "调用分割预测接口")
        
        # 检查测试图像
        if not Path(TEST_IMAGE_PATH).exists():
            print(f"❌ 测试图像不存在: {TEST_IMAGE_PATH}")
            print(f"\n请提供测试图像，或修改 TEST_IMAGE_PATH 变量")
            print(f"示例：TEST_IMAGE_PATH = './uploads/sample.png'")
            return
        
        # 调用预测接口
        result = predict_segmentation(TEST_IMAGE_PATH, weight_id)
        
        # ==================== 步骤3：解码并保存掩码 ====================
        print_step(3, "解码Base64掩码并保存")
        
        mask_base64 = result.get("mask_base64")
        if not mask_base64:
            print("❌ 响应中未包含mask_base64字段")
            print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return
        
        output_mask_path = OUTPUT_DIR / "api_result_mask.png"
        api_mask = decode_and_save_mask(mask_base64, str(output_mask_path))
        
        # ==================== 步骤4：与本地推理对比（可选）====================
        print_step(4, "与本地推理结果对比（可选）")
        compare_with_local_inference(api_mask, TEST_IMAGE_PATH)
        
        # ==================== 步骤5：生成可视化对比图 ====================
        print_step(5, "生成可视化对比图")
        visualize_result(TEST_IMAGE_PATH, str(output_mask_path))
        
        # ==================== 测试总结 ====================
        print_section("✅ 联调测试完成")
        
        print("\n📊 测试结果总结:")
        print(f"  ✓ 健康检查: 通过")
        print(f"  ✓ 权重管理: {'上传成功' if weight_id else '使用官方权重'}")
        print(f"  ✓ 分割预测: 成功")
        print(f"  ✓ 掩码解码: 成功")
        print(f"  ✓ 推理设备: {result.get('device')}")
        print(f"  ✓ 推理耗时: {result.get('infer_time')}秒")
        
        print("\n📁 输出文件:")
        print(f"  - 分割掩码: {output_mask_path}")
        print(f"  - 对比图: {OUTPUT_DIR / 'comparison.png'}")
        if (OUTPUT_DIR / "local_result.png").exists():
            print(f"  - 本地结果: {OUTPUT_DIR / 'local_result.png'}")
        if (OUTPUT_DIR / "diff_mask.png").exists():
            print(f"  - 差异图: {OUTPUT_DIR / 'diff_mask.png'}")
        
        print("\n💡 后续步骤:")
        print("  1. 查看输出文件验证分割效果")
        print("  2. 检查日志文件查看详细执行信息")
        print("  3. 使用浏览器访问 http://127.0.0.1:8000/docs 查看API文档")
        print("  4. 集成到前端进行端到端测试")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 故障排查建议:")
        print("  1. 检查后端服务是否正常运行")
        print("  2. 检查测试文件路径是否正确")
        print("  3. 查看后端日志 ./logs/octa_backend.log")
        print("  4. 确认网络连接正常")


# ==================== 快速测试函数 ====================

def quick_test_official_weight():
    """快速测试：使用官方权重进行单次预测"""
    
    print_section("快速测试：官方权重预测")
    
    # 检查服务
    if not check_server_health():
        print("❌ 后端服务未启动")
        return
    
    # 检查测试图像
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ 测试图像不存在: {TEST_IMAGE_PATH}")
        return
    
    try:
        # 预测
        print("\n🔍 正在预测...")
        result = predict_segmentation(TEST_IMAGE_PATH, weight_id=None)
        
        # 保存结果
        mask_base64 = result.get("mask_base64")
        output_path = OUTPUT_DIR / "quick_test_result.png"
        decode_and_save_mask(mask_base64, str(output_path))
        
        print(f"\n✅ 快速测试完成！")
        print(f"   结果已保存: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 快速测试失败: {e}")


# ==================== 程序入口 ====================

if __name__ == "__main__":
    """
    运行方式：
    
    1. 完整测试（包含权重上传、对比等）:
        python test_seg_api.py
    
    2. 快速测试（仅测试预测功能）:
        python -c "from test_seg_api import quick_test_official_weight; quick_test_official_weight()"
    
    3. 交互式测试（Python REPL）:
        python
        >>> from test_seg_api import *
        >>> check_server_health()
        >>> result = predict_segmentation("./test.png")
    """
    
    # 运行完整测试
    main()
    
    # 或运行快速测试（取消下行注释）
    # quick_test_official_weight()
