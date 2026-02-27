"""
文件管理DAO测试脚本

本脚本用于测试file_dao.py中所有CRUD功能的正确性。

测试内容：
1. 数据库表创建
2. 添加文件记录（图片和数据集）
3. 查询所有文件
4. 按类型筛选查询
5. 按ID查询单个文件
6. 更新文件关联模型
7. 删除文件记录和本地文件

运行方式：
    cd octa_backend
    python test_file_dao.py
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from dao.file_dao import FileDAO


def print_separator(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}".center(70))
    print("=" * 70)


def test_create_table():
    """测试1：创建数据库表"""
    print_separator("测试1：创建数据库表")
    
    success = FileDAO.create_file_table()
    
    if success:
        print("✓ 数据库表创建成功")
    else:
        print("✗ 数据库表创建失败")
    
    return success


def test_add_records():
    """测试2：添加文件记录"""
    print_separator("测试2：添加文件记录")
    
    # 添加图片记录
    print("\n[2.1] 添加图片记录...")
    image_id = FileDAO.add_file_record(
        file_name='octa_sample_001.png',
        file_path='uploads/images/octa_sample_001.png',
        file_type='image',
        file_size=2.5
    )
    
    if image_id:
        print(f"✓ 图片记录添加成功，ID: {image_id}")
    else:
        print("✗ 图片记录添加失败")
    
    # 添加数据集记录
    print("\n[2.2] 添加数据集记录...")
    dataset_id = FileDAO.add_file_record(
        file_name='training_dataset.zip',
        file_path='uploads/datasets/training_dataset.zip',
        file_type='dataset',
        related_model='models/weights/unet_octa.pth',
        file_size=120.8
    )
    
    if dataset_id:
        print(f"✓ 数据集记录添加成功，ID: {dataset_id}")
    else:
        print("✗ 数据集记录添加失败")
    
    # 再添加一条图片记录
    print("\n[2.3] 添加第二条图片记录...")
    image_id2 = FileDAO.add_file_record(
        file_name='octa_sample_002.png',
        file_path='uploads/images/octa_sample_002.png',
        file_type='image',
        file_size=3.2
    )
    
    if image_id2:
        print(f"✓ 第二条图片记录添加成功，ID: {image_id2}")
    else:
        print("✗ 第二条图片记录添加失败")
    
    return image_id, dataset_id, image_id2


def test_get_all_files():
    """测试3：查询所有文件"""
    print_separator("测试3：查询所有文件")
    
    all_files = FileDAO.get_file_list()
    
    if all_files:
        print(f"\n✓ 查询成功，共找到 {len(all_files)} 条记录\n")
        
        # 打印表头
        print(f"{'ID':<5} {'文件名':<30} {'类型':<10} {'大小(MB)':<12} {'上传时间':<20}")
        print("-" * 80)
        
        # 打印每条记录
        for file in all_files:
            file_size = f"{file['file_size']:.2f}" if file['file_size'] else "N/A"
            print(f"{file['id']:<5} {file['file_name']:<30} {file['file_type']:<10} "
                  f"{file_size:<12} {file['upload_time']:<20}")
    else:
        print("✗ 查询失败或无记录")
    
    return all_files


def test_get_files_by_type():
    """测试4：按类型查询文件"""
    print_separator("测试4：按类型查询文件")
    
    # 查询图片文件
    print("\n[4.1] 查询所有图片文件...")
    images = FileDAO.get_file_list(file_type='image')
    
    if images:
        print(f"✓ 找到 {len(images)} 个图片文件")
        for img in images:
            print(f"  - {img['file_name']}: {img['file_size']} MB")
    else:
        print("✗ 未找到图片文件")
    
    # 查询数据集文件
    print("\n[4.2] 查询所有数据集文件...")
    datasets = FileDAO.get_file_list(file_type='dataset')
    
    if datasets:
        print(f"✓ 找到 {len(datasets)} 个数据集文件")
        for ds in datasets:
            print(f"  - {ds['file_name']}: {ds['file_size']} MB")
            if ds['related_model']:
                print(f"    关联模型: {ds['related_model']}")
    else:
        print("✗ 未找到数据集文件")
    
    return images, datasets


def test_get_file_by_id(file_id):
    """测试5：按ID查询文件"""
    print_separator(f"测试5：按ID查询文件（ID={file_id}）")
    
    file_info = FileDAO.get_file_by_id(file_id)
    
    if file_info:
        print(f"\n✓ 查询成功\n")
        print(f"文件ID:      {file_info['id']}")
        print(f"文件名:      {file_info['file_name']}")
        print(f"文件路径:    {file_info['file_path']}")
        print(f"文件类型:    {file_info['file_type']}")
        print(f"上传时间:    {file_info['upload_time']}")
        print(f"关联模型:    {file_info['related_model'] or 'N/A'}")
        print(f"文件大小:    {file_info['file_size']} MB" if file_info['file_size'] else "文件大小:    N/A")
    else:
        print("✗ 查询失败或文件不存在")
    
    return file_info


def test_update_relation(file_id):
    """测试6：更新文件关联模型"""
    print_separator(f"测试6：更新文件关联模型（ID={file_id}）")
    
    new_model = 'models/weights/unet_trained_20260116.pth'
    
    success = FileDAO.update_file_relation(file_id, new_model)
    
    if success:
        print(f"✓ 模型关联更新成功")
        print(f"  新模型路径: {new_model}")
        
        # 验证更新结果
        print("\n验证更新结果...")
        file_info = FileDAO.get_file_by_id(file_id)
        if file_info and file_info['related_model'] == new_model:
            print("✓ 验证成功，模型路径已更新")
        else:
            print("✗ 验证失败，模型路径未更新")
    else:
        print("✗ 模型关联更新失败")
    
    return success


def test_delete_file(file_id):
    """测试7：删除文件"""
    print_separator(f"测试7：删除文件（ID={file_id}）")
    
    # 先查询文件信息
    file_info = FileDAO.get_file_by_id(file_id)
    if file_info:
        print(f"\n准备删除文件: {file_info['file_name']}")
        print(f"文件路径: {file_info['file_path']}")
    
    # 执行删除
    success = FileDAO.delete_file(file_id)
    
    if success:
        print(f"\n✓ 文件删除成功")
        
        # 验证删除结果
        print("\n验证删除结果...")
        file_info_after = FileDAO.get_file_by_id(file_id)
        if not file_info_after:
            print("✓ 验证成功，记录已从数据库删除")
        else:
            print("✗ 验证失败，记录仍存在于数据库")
    else:
        print(f"✗ 文件删除失败")
    
    return success


def test_invalid_operations():
    """测试8：异常情况处理"""
    print_separator("测试8：异常情况处理")
    
    # 测试无效文件类型
    print("\n[8.1] 测试无效文件类型...")
    invalid_id = FileDAO.add_file_record(
        file_name='invalid.txt',
        file_path='uploads/invalid.txt',
        file_type='invalid_type',  # 无效类型
        file_size=1.0
    )
    
    if not invalid_id:
        print("✓ 正确拒绝了无效文件类型")
    else:
        print("✗ 应该拒绝无效文件类型")
    
    # 测试查询不存在的ID
    print("\n[8.2] 测试查询不存在的ID...")
    non_exist = FileDAO.get_file_by_id(99999)
    
    if not non_exist:
        print("✓ 正确处理了不存在的ID")
    else:
        print("✗ 应该返回None")
    
    # 测试删除不存在的ID
    print("\n[8.3] 测试删除不存在的ID...")
    delete_result = FileDAO.delete_file(99999)
    
    if not delete_result:
        print("✓ 正确处理了不存在的ID删除")
    else:
        print("✗ 应该返回False")


def main():
    """主测试流程"""
    print("\n" + "=" * 70)
    print("  OCTA文件管理DAO测试脚本".center(70))
    print("=" * 70)
    
    try:
        # 测试1：创建表
        if not test_create_table():
            print("\n[ERROR] 数据库表创建失败，终止测试")
            return
        
        # 测试2：添加记录
        image_id, dataset_id, image_id2 = test_add_records()
        
        if not image_id or not dataset_id or not image_id2:
            print("\n[ERROR] 添加记录失败，终止测试")
            return
        
        # 测试3：查询所有文件
        all_files = test_get_all_files()
        
        # 测试4：按类型查询
        images, datasets = test_get_files_by_type()
        
        # 测试5：按ID查询
        test_get_file_by_id(image_id)
        
        # 测试6：更新关联
        test_update_relation(image_id)
        
        # 测试7：删除文件（删除第一个图片）
        test_delete_file(image_id)
        
        # 测试8：异常情况
        test_invalid_operations()
        
        # 最终状态
        print_separator("测试完成 - 最终数据库状态")
        final_files = FileDAO.get_file_list()
        print(f"\n剩余文件数: {len(final_files)}")
        
        print("\n" + "=" * 70)
        print("  测试完成！".center(70))
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
