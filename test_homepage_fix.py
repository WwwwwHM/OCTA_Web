"""
首页功能完整性测试脚本
验证级联下拉功能是否正常工作

测试内容：
1. 前端服务运行状态
2. 后端服务运行状态  
3. /file/model-weights API端点
4. watch导入是否正确
"""

import requests
import time

def test_frontend():
    """测试前端服务"""
    print("=" * 70)
    print("测试1: 前端服务状态")
    print("=" * 70)
    try:
        response = requests.get("http://localhost:5174", timeout=3)
        if response.status_code == 200:
            print("✅ 前端服务正常运行 (端口5174)")
            return True
        else:
            print(f"❌ 前端返回异常状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到前端服务 (http://localhost:5174)")
        print("   请运行: cd octa_frontend && npm run dev")
        return False
    except Exception as e:
        print(f"❌ 前端测试失败: {e}")
        return False


def test_backend():
    """测试后端服务"""
    print("\n" + "=" * 70)
    print("测试2: 后端服务状态")
    print("=" * 70)
    try:
        response = requests.get("http://127.0.0.1:8000", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print("✅ 后端服务正常运行")
            print(f"   消息: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ 后端返回异常状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务 (http://127.0.0.1:8000)")
        print("   请运行: cd octa_backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ 后端测试失败: {e}")
        return False


def test_model_weights_api():
    """测试新的model-weights API端点"""
    print("\n" + "=" * 70)
    print("测试3: /file/model-weights API端点")
    print("=" * 70)
    
    test_cases = [
        ("unet", "U-Net"),
        ("rs_unet3_plus", "RS-Unet3+"),
        (None, "无参数")
    ]
    
    success_count = 0
    
    for model_type, display_name in test_cases:
        try:
            if model_type:
                url = f"http://127.0.0.1:8000/file/model-weights?model_type={model_type}"
            else:
                url = "http://127.0.0.1:8000/file/model-weights"
            
            print(f"\n测试 {display_name} 权重查询:")
            print(f"  URL: {url}")
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 200:
                    weight_count = len(data.get('data', []))
                    print(f"  ✅ 查询成功: {data.get('msg')}")
                    print(f"  📦 权重数量: {weight_count}")
                    success_count += 1
                else:
                    print(f"  ⚠️ 业务状态码异常: {data.get('code')}")
            elif response.status_code == 400:
                print(f"  ✅ 参数验证正常 (预期400)")
                success_count += 1
            else:
                print(f"  ❌ HTTP状态码异常: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ 请求失败: {e}")
    
    print(f"\n总计: {success_count}/{len(test_cases)} 测试通过")
    return success_count == len(test_cases)


def test_frontend_code():
    """检查前端代码是否正确导入watch"""
    print("\n" + "=" * 70)
    print("测试4: 前端代码检查")
    print("=" * 70)
    
    file_path = "d:\\Code\\OCTA_Web\\octa_frontend\\src\\views\\HomeView.vue"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查watch导入
        if "import { ref, onMounted, computed, watch } from 'vue'" in content:
            print("✅ watch已正确导入")
        else:
            print("❌ watch未导入或导入不正确")
            return False
        
        # 检查watch监听器
        if "watch(selectedModel," in content:
            print("✅ watch监听器已定义")
        else:
            print("❌ watch监听器未找到")
            return False
        
        # 检查fetchWeights函数
        if "const fetchWeights = async (modelType = null)" in content:
            print("✅ fetchWeights函数已更新")
        else:
            print("❌ fetchWeights函数未找到或未更新")
            return False
        
        # 检查API调用
        if "/file/model-weights?model_type=" in content:
            print("✅ API端点调用正确")
        else:
            print("❌ API端点调用有误")
            return False
        
        return True
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return False
    except Exception as e:
        print(f"❌ 文件检查失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print(" 首页功能完整性测试".center(70))
    print("=" * 70)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {
        "前端服务": test_frontend(),
        "后端服务": test_backend(),
        "API端点": test_model_weights_api(),
        "前端代码": test_frontend_code()
    }
    
    # 总结
    print("\n" + "=" * 70)
    print(" 测试总结".center(70))
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print("=" * 70)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！首页功能正常")
        print("\n访问地址:")
        print("  前端: http://localhost:5174")
        print("  后端: http://127.0.0.1:8000")
        print("  API文档: http://127.0.0.1:8000/docs")
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误信息")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
