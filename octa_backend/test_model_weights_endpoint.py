"""
测试 /file/model-weights 端点功能
验证按模型类型筛选权重文件的API

测试用例：
1. 无参数请求 → 返回空列表
2. 有效模型类型（unet、fcn、rs_unet3_plus） → 返回对应权重
3. 无效模型类型 → 返回400错误
4. 端点响应格式验证
"""

import requests
import json
from typing import Dict, Any


# ==================== 配置 ====================
BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/file/model-weights"


def test_endpoint(params: Dict[str, Any] = None, test_name: str = ""):
    """
    测试端点的通用函数
    
    Args:
        params: 查询参数字典
        test_name: 测试用例名称
    """
    print(f"\n{'='*70}")
    print(f"测试用例: {test_name}")
    print(f"{'='*70}")
    
    # 构建完整URL
    url = f"{BASE_URL}{ENDPOINT}"
    if params:
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{url}?{param_str}"
    
    print(f"请求URL: {url}")
    
    try:
        # 发送GET请求
        response = requests.get(url, timeout=10)
        
        # 打印响应状态码
        print(f"响应状态码: {response.status_code}")
        
        # 解析JSON响应
        try:
            data = response.json()
            print(f"响应格式: JSON")
            print(f"响应内容:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # 验证响应结构
            if "code" in data and "msg" in data and "data" in data:
                print(f"✓ 响应格式符合标准（包含code、msg、data字段）")
            else:
                print(f"✗ 响应格式不符合标准")
            
            # 验证响应码
            if response.status_code == 200:
                print(f"✓ HTTP状态码正确")
                if data.get("code") == 200:
                    print(f"✓ 业务状态码正确")
                    print(f"✓ 找到 {len(data.get('data', []))} 个权重文件")
                else:
                    print(f"✗ 业务状态码错误: {data.get('code')}")
            elif response.status_code == 400:
                print(f"✓ 参数错误正确返回400状态码")
                print(f"错误信息: {data.get('detail', 'N/A')}")
            else:
                print(f"✗ 意外的HTTP状态码")
                
        except json.JSONDecodeError:
            print(f"✗ 响应不是有效的JSON格式")
            print(f"原始响应: {response.text[:200]}")
        
    except requests.exceptions.ConnectionError:
        print(f"✗ 无法连接到后端服务器")
        print(f"请确保后端服务正在运行：python main.py")
    except requests.exceptions.Timeout:
        print(f"✗ 请求超时（超过10秒）")
    except Exception as e:
        print(f"✗ 请求失败: {e}")


def main():
    """运行所有测试用例"""
    print("=" * 70)
    print(" /file/model-weights 端点功能测试".center(70))
    print("=" * 70)
    print(f"目标服务器: {BASE_URL}")
    print(f"测试端点: {ENDPOINT}")
    print("=" * 70)
    
    # 测试用例1：无参数（应返回空列表）
    test_endpoint(
        params=None,
        test_name="测试1 - 无参数请求（应提示选择模型类型）"
    )
    
    # 测试用例2：有效模型类型 - unet
    test_endpoint(
        params={"model_type": "unet"},
        test_name="测试2 - 查询U-Net模型权重"
    )
    
    # 测试用例3：有效模型类型 - rs_unet3_plus
    test_endpoint(
        params={"model_type": "rs_unet3_plus"},
        test_name="测试3 - 查询RS-Unet3+模型权重"
    )
    
    # 测试用例4：无效模型类型（应返回400错误）
    test_endpoint(
        params={"model_type": "invalid_model"},
        test_name="测试4 - 无效模型类型（应返回400错误）"
    )
    
    # 测试用例5：空字符串模型类型（应返回400错误）
    test_endpoint(
        params={"model_type": ""},
        test_name="测试6 - 空字符串模型类型（应返回400错误）"
    )
    
    # 总结
    print(f"\n{'='*70}")
    print(" 测试完成".center(70))
    print("=" * 70)
    print("\n使用说明：")
    print("1. 确保后端服务正在运行（python main.py）")
    print("2. 确保数据库已迁移（python migrate_add_model_type.py）")
    print("3. 如需添加测试数据，可以先上传权重文件到文件管理系统")
    print("4. 前端集成时，使用以下方式调用：")
    print("   axios.get(`http://127.0.0.1:8000/file/model-weights?model_type=${model}`)")
    print("=" * 70)


if __name__ == "__main__":
    main()
