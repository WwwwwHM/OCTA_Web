"""
OCTA 数据库功能测试脚本

此脚本用于验证SQLite数据库的正确性和各个接口的功能。
可在应用启动后运行此脚本进行完整功能测试。

使用方法：
    python test_database.py

依赖：
    - FastAPI应用已启动在 http://127.0.0.1:8000
    - requests 库已安装
"""

import requests
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict


# ==================== 配置 ====================

API_BASE_URL = "http://127.0.0.1:8000"
DB_PATH = Path("./octa.db")

# ANSI颜色代码用于终端输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


# ==================== 数据库直接检查 ====================

def check_database_file() -> bool:
    """
    检查数据库文件是否存在
    
    Returns:
        True if database exists, False otherwise
    """
    print(f"\n{Colors.BLUE}[检查] 数据库文件是否存在...{Colors.RESET}")
    
    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"{Colors.GREEN}✓ 数据库文件已找到{Colors.RESET}")
        print(f"  路径: {DB_PATH.absolute()}")
        print(f"  大小: {size_mb:.2f} MB")
        return True
    else:
        print(f"{Colors.RED}✗ 数据库文件不存在{Colors.RESET}")
        print(f"  预期路径: {DB_PATH.absolute()}")
        return False


def check_database_table() -> bool:
    """
    检查images表是否存在及其结构
    
    Returns:
        True if table exists and has correct structure, False otherwise
    """
    print(f"\n{Colors.BLUE}[检查] 数据库表结构...{Colors.RESET}")
    
    if not DB_PATH.exists():
        print(f"{Colors.RED}✗ 数据库文件不存在，无法检查表{Colors.RESET}")
        return False
    
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
        cursor = conn.cursor()
        
        # 查询表信息
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='images'")
        result = cursor.fetchone()
        
        if result is None:
            print(f"{Colors.RED}✗ images表不存在{Colors.RESET}")
            return False
        
        print(f"{Colors.GREEN}✓ images表已创建{Colors.RESET}")
        print(f"\n表结构:")
        print(f"  {result[0]}")
        
        # 查询行数
        cursor.execute("SELECT COUNT(*) FROM images")
        count = cursor.fetchone()[0]
        print(f"\n当前记录数: {count}")
        
        # 查询列信息
        cursor.execute("PRAGMA table_info(images)")
        columns = cursor.fetchall()
        
        print(f"\n列详情:")
        for col_id, col_name, col_type, not_null, default, pk in columns:
            pk_mark = " [PK]" if pk else ""
            null_mark = " [NOT NULL]" if not_null else ""
            print(f"  {col_name}: {col_type}{pk_mark}{null_mark}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ 检查表结构失败: {e}{Colors.RESET}")
        return False


def get_database_records() -> Optional[List[Dict]]:
    """
    直接从数据库查询所有记录
    
    Returns:
        List of records or None if failed
    """
    print(f"\n{Colors.BLUE}[检查] 数据库中的记录...{Colors.RESET}")
    
    if not DB_PATH.exists():
        print(f"{Colors.RED}✗ 数据库文件不存在{Colors.RESET}")
        return None
    
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM images ORDER BY upload_time DESC")
        rows = cursor.fetchall()
        records = [dict(row) for row in rows]
        
        conn.close()
        
        if records:
            print(f"{Colors.GREEN}✓ 查询到 {len(records)} 条记录{Colors.RESET}")
            print(f"\n记录详情:")
            for i, record in enumerate(records[:5], 1):  # 显示前5条
                print(f"\n  [{i}] ID: {record['id']}")
                print(f"      文件名: {record['filename']}")
                print(f"      时间: {record['upload_time']}")
                print(f"      模型: {record['model_type']}")
            if len(records) > 5:
                print(f"\n  ... 还有 {len(records) - 5} 条记录")
        else:
            print(f"{Colors.YELLOW}! 数据库中暂无记录{Colors.RESET}")
        
        return records
        
    except Exception as e:
        print(f"{Colors.RED}✗ 查询记录失败: {e}{Colors.RESET}")
        return None


# ==================== API接口测试 ====================

def test_api_connectivity() -> bool:
    """
    测试API服务是否可访问
    
    Returns:
        True if API is accessible, False otherwise
    """
    print(f"\n{Colors.BLUE}[API测试] 检查服务连接...{Colors.RESET}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"{Colors.GREEN}✓ 后端服务运行正常{Colors.RESET}")
            print(f"  响应: {data['message']}")
            return True
        else:
            print(f"{Colors.RED}✗ 服务返回异常状态码: {response.status_code}{Colors.RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}✗ 无法连接到后端服务{Colors.RESET}")
        print(f"  请确保服务运行在 {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ 连接测试失败: {e}{Colors.RESET}")
        return False


def test_history_api() -> bool:
    """
    测试 GET /history/ 接口
    
    Returns:
        True if API works correctly, False otherwise
    """
    print(f"\n{Colors.BLUE}[API测试] GET /history/ - 获取所有历史记录{Colors.RESET}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/history/", timeout=5)
        
        if response.status_code == 200:
            records = response.json()
            if isinstance(records, list):
                print(f"{Colors.GREEN}✓ 接口返回正常 (200 OK){Colors.RESET}")
                print(f"  返回记录数: {len(records)}")
                
                if records:
                    print(f"\n  前3条记录:")
                    for i, record in enumerate(records[:3], 1):
                        print(f"\n  [{i}]")
                        print(f"      ID: {record.get('id', 'N/A')}")
                        print(f"      文件名: {record.get('filename', 'N/A')}")
                        print(f"      时间: {record.get('upload_time', 'N/A')}")
                        print(f"      模型: {record.get('model_type', 'N/A')}")
                else:
                    print(f"  (暂无记录)")
                
                return True
            else:
                print(f"{Colors.RED}✗ 响应格式错误，应该是列表{Colors.RESET}")
                return False
        else:
            print(f"{Colors.RED}✗ 接口返回错误状态码: {response.status_code}{Colors.RESET}")
            print(f"  响应: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}✗ 接口测试失败: {e}{Colors.RESET}")
        return False


def test_history_detail_api() -> bool:
    """
    测试 GET /history/{id} 接口
    
    Returns:
        True if API works correctly, False otherwise
    """
    print(f"\n{Colors.BLUE}[API测试] GET /history/{{id}} - 获取单条记录{Colors.RESET}")
    
    # 首先获取最新的记录ID
    try:
        response = requests.get(f"{API_BASE_URL}/history/", timeout=5)
        if response.status_code != 200:
            print(f"{Colors.YELLOW}! 跳过此测试 (无法获取记录列表){Colors.RESET}")
            return True
        
        records = response.json()
        if not records:
            print(f"{Colors.YELLOW}! 跳过此测试 (数据库中暂无记录){Colors.RESET}")
            return True
        
        test_id = records[0]['id']
        
        # 测试有效的ID
        print(f"\n  [测试1] 查询存在的记录 (ID: {test_id})")
        response = requests.get(f"{API_BASE_URL}/history/{test_id}", timeout=5)
        
        if response.status_code == 200:
            record = response.json()
            print(f"  {Colors.GREEN}✓ 记录查询成功 (200 OK){Colors.RESET}")
            print(f"    ID: {record.get('id')}")
            print(f"    文件名: {record.get('filename')}")
            print(f"    时间: {record.get('upload_time')}")
            print(f"    模型: {record.get('model_type')}")
        else:
            print(f"  {Colors.RED}✗ 查询失败: {response.status_code}{Colors.RESET}")
            return False
        
        # 测试不存在的ID
        print(f"\n  [测试2] 查询不存在的记录 (ID: 99999)")
        response = requests.get(f"{API_BASE_URL}/history/99999", timeout=5)
        
        if response.status_code == 404:
            error = response.json()
            print(f"  {Colors.GREEN}✓ 返回正确的404错误{Colors.RESET}")
            print(f"    错误信息: {error.get('detail')}")
        else:
            print(f"  {Colors.RED}✗ 应该返回404，但返回: {response.status_code}{Colors.RESET}")
            return False
        
        print(f"\n{Colors.GREEN}✓ 接口测试全部通过{Colors.RESET}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ 接口测试失败: {e}{Colors.RESET}")
        return False


def test_invalid_parameters() -> bool:
    """
    测试参数验证
    
    Returns:
        True if parameter validation works, False otherwise
    """
    print(f"\n{Colors.BLUE}[API测试] 参数验证...{Colors.RESET}")
    
    print(f"\n  [测试] 无效的记录ID (ID: -1)")
    try:
        response = requests.get(f"{API_BASE_URL}/history/-1", timeout=5)
        
        if response.status_code == 400:
            error = response.json()
            print(f"  {Colors.GREEN}✓ 返回正确的400错误{Colors.RESET}")
            print(f"    错误信息: {error.get('detail')}")
            return True
        else:
            print(f"  {Colors.RED}✗ 应该返回400，但返回: {response.status_code}{Colors.RESET}")
            return False
    except Exception as e:
        print(f"  {Colors.RED}✗ 测试失败: {e}{Colors.RESET}")
        return False


# ==================== 摘要报告 ====================

def print_summary(results: Dict[str, bool]):
    """
    打印测试摘要
    
    Args:
        results: 测试结果字典
    """
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}测试摘要{Colors.RESET}")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✓ 通过{Colors.RESET}" if result else f"{Colors.RED}✗ 失败{Colors.RESET}"
        print(f"{test_name:<40} {status}")
    
    print(f"\n{Colors.BOLD}总体结果: {passed}/{total} 测试通过{Colors.RESET}")
    
    if failed == 0:
        print(f"{Colors.GREEN}✓ 所有测试通过！{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ 有 {failed} 个测试失败{Colors.RESET}")


# ==================== 主测试流程 ====================

def main():
    """
    运行所有测试
    """
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}OCTA 数据库功能测试{Colors.RESET}")
    print(f"{'='*60}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API 地址: {API_BASE_URL}")
    print(f"数据库位置: {DB_PATH.absolute()}")
    
    results = {}
    
    # ==================== 数据库检查 ====================
    print(f"\n{Colors.BOLD}[第1部分] 数据库文件检查{Colors.RESET}")
    results["1. 数据库文件存在"] = check_database_file()
    results["2. 数据库表结构"] = check_database_table()
    results["3. 数据库记录检查"] = get_database_records() is not None
    
    # ==================== API连接检查 ====================
    print(f"\n{Colors.BOLD}[第2部分] API 连接检查{Colors.RESET}")
    api_available = test_api_connectivity()
    results["4. API 服务连接"] = api_available
    
    if not api_available:
        print(f"\n{Colors.YELLOW}! API 服务不可用，跳过接口测试{Colors.RESET}")
        print(f"  请确保在另一个终端运行: cd octa_backend && python main.py")
    else:
        # ==================== API接口测试 ====================
        print(f"\n{Colors.BOLD}[第3部分] API 接口功能测试{Colors.RESET}")
        results["5. /history/ 接口"] = test_history_api()
        results["6. /history/{{id}} 接口"] = test_history_detail_api()
        results["7. 参数验证"] = test_invalid_parameters()
    
    # ==================== 打印摘要 ====================
    print_summary(results)
    
    # ==================== 返回状态码 ====================
    all_passed = all(results.values())
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}! 测试被用户中断{Colors.RESET}")
        exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}✗ 测试过程中发生意外错误: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        exit(1)

