"""OCTA图像分割平台 - 文件工具类（File Utilities）

本模块实现了FileUtils类，专门负责与文件系统相关的所有操作。

架构设计理念：
  1. 文件操作隔离：所有文件I/O、验证、保存集中在这里
  2. 接口清晰：提供明确的文件操作接口，隐藏实现细节
  3. 验证全面：支持格式、大小等多维度文件校验
  4. 错误处理：所有文件操作都有完整的异常捕获和错误提示
  5. 易于扩展：添加新的文件操作方法无需改Controller

文件处理流程：
  1. 验证文件格式（格式白名单）
  2. 验证文件大小（大小限制）
  3. 生成唯一文件名（UUID+原后缀）
  4. 创建存储目录（自动创建不存在的目录）
  5. 保存文件到磁盘（error handling）
  6. 返回保存状态和文件路径

作者：OCTA Web项目组
日期：2026年1月14日
"""

import os
import uuid
from pathlib import Path
from typing import Tuple, List, Optional

# 导入配置
from config.config import ALLOWED_FORMATS, MAX_FILE_SIZE


class FileUtils:
    """
    文件处理工具类
    
    负责所有与文件系统相关的操作，包括：
    - 文件格式验证
    - 文件大小验证
    - 唯一文件名生成
    - 目录创建和管理
    - 文件保存
    
    所有方法均为静态方法，无需实例化：
    
    使用示例：
        >>> # 验证文件格式
        >>> is_valid, error_msg = FileUtils.validate_file_format('image.png')
        >>> if not is_valid:
        ...     print(f"文件格式错误: {error_msg}")
        
        >>> # 验证文件大小（10MB）
        >>> is_valid, error_msg = FileUtils.validate_file_size(file_obj, max_size=10*1024*1024)
        
        >>> # 生成唯一文件名
        >>> unique_name = FileUtils.generate_unique_filename('photo.jpg')
        >>> print(unique_name)  # img_abc123def456.jpg
        
        >>> # 创建目录
        >>> FileUtils.create_dir_if_not_exists('uploads/')
        
        >>> # 保存文件
        >>> success, message = FileUtils.save_uploaded_file(file_obj, 'uploads/img.png')
    """
    
    # ==================== 常量定义 ====================
    
    # 默认允许的文件格式（从config加载）
    DEFAULT_ALLOWED_FORMATS = ALLOWED_FORMATS
    
    # 默认最大文件大小（从config加载）
    DEFAULT_MAX_FILE_SIZE = MAX_FILE_SIZE
    
    # ==================== 格式校验 ====================
    
    @staticmethod
    def validate_file_format(
        filename: str,
        allow_formats: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        验证文件格式是否被允许
        
        功能说明：
          - 从文件名中提取扩展名
          - 与白名单格式对比（大小写不敏感）
          - 返回验证结果和错误提示
        
        实现细节：
          - 文件名为空：返回False
          - 无扩展名：返回False
          - 格式验证时不区分大小写：.JPG/.png都被接受
          - 格式为小写进行比较：jpg == JPG
        
        Args:
            filename (str): 
              待验证的文件名（不是完整路径）
              示例：'photo.jpg'、'image.PNG'、'document.txt'
            
            allow_formats (List[str], optional): 
              允许的格式列表（无需包含点符号）
              默认：ALLOWED_FORMATS（从配置加载）
              示例：['png', 'jpg', 'bmp']
              注意：格式应为小写，函数会自动转换比较
        
        Returns:
            Tuple[bool, str]: 
              返回元组包含：
              - 第1个元素：是否有效（True/False）
              - 第2个元素：错误提示信息或成功提示
              
              返回示例：
                (True, "✓ 文件格式有效")
                (False, "✗ 不支持的文件格式: txt，仅支持: [from ALLOWED_FORMATS]")
        
        异常场景（已处理）：
          - filename为空：返回(False, "文件名为空")
          - 无扩展名：返回(False, "文件名无扩展名")
          - 格式不支持：返回(False, "不支持的格式")
          - 格式为None：返回(False, "无法识别格式")
        
        示例:
            >>> # 有效的格式
            >>> is_valid, msg = FileUtils.validate_file_format('image.png')
            >>> print(is_valid)  # True
            
            >>> # 无效的格式
            >>> is_valid, msg = FileUtils.validate_file_format('image.gif')
            >>> print(is_valid)  # False
            
            >>> # 大小写不敏感
            >>> is_valid, msg = FileUtils.validate_file_format('image.JPG')
            >>> print(is_valid)  # True
            
            >>> # 自定义格式列表
            >>> is_valid, msg = FileUtils.validate_file_format(
            ...     'document.pdf',
            ...     allow_formats=['pdf', 'doc', 'docx']
            ... )
        """
        try:
            # ==================== 步骤1：参数验证 ====================
            # 检查filename是否为空或无效
            if not filename or not isinstance(filename, str):
                return (False, "✗ 文件名为空或格式无效")
            
            # ==================== 步骤2：使用默认格式白名单 ====================
            # 如果用户未指定，使用默认的允许格式列表
            if allow_formats is None:
                allow_formats = FileUtils.DEFAULT_ALLOWED_FORMATS
            
            # 将允许格式转换为小写（便于不区分大小写的比较）
            allow_formats_lower = [fmt.lower() for fmt in allow_formats]
            
            # ==================== 步骤3：提取文件扩展名 ====================
            # 使用os.path.splitext()分离文件名和扩展名
            # splitext返回元组：('文件名', '.扩展名')
            # 例如：'image.png' -> ('image', '.png')
            _, file_extension = os.path.splitext(filename)
            
            # 检查是否有扩展名
            if not file_extension:
                return (False, "✗ 文件名无扩展名")
            
            # 移除扩展名的点符号并转为小写
            # '.png' -> 'png'
            file_format = file_extension.lstrip('.').lower()
            
            # ==================== 步骤4：格式白名单检查 ====================
            # 检查文件格式是否在允许的格式列表中
            if file_format not in allow_formats_lower:
                # 格式不支持，返回错误信息和支持的格式列表
                supported_formats = ', '.join(allow_formats_lower)
                return (
                    False,
                    f"✗ 不支持的文件格式: {file_format}，仅支持: {supported_formats}"
                )
            
            # ==================== 步骤5：格式验证成功 ====================
            return (True, f"✓ 文件格式有效: {file_format.upper()}")
            
        except Exception as e:
            # 捕获所有异常（如路径解析错误等）
            return (False, f"✗ 文件格式验证失败: {str(e)}")
    
    # ==================== 大小校验 ====================
    
    @staticmethod
    def validate_file_size(
        file_obj,
        max_size: int = None
    ) -> Tuple[bool, str]:
        """
        验证上传文件的大小是否超过限制
        
        功能说明：
          - 获取文件大小（单位：字节）
          - 与限制大小对比
          - 返回验证结果和友好的错误提示
        
        实现细节：
          - 支持多种类型的file_obj（UploadFile、文件对象等）
          - 自动计算并显示文件大小（字节/MB转换）
          - 错误提示包含文件实际大小和限制大小
        
        Args:
            file_obj: 
              上传的文件对象
              支持类型：
                - FastAPI的UploadFile对象
                - Python的文件对象
                - 其他具有.file属性或.seek()方法的对象
            
            max_size (int, optional): 
              最大允许文件大小，单位为字节（Bytes）
              默认：MAX_FILE_SIZE（从配置加载）
              
              大小单位参考：
                - 1 KB = 1024 Bytes
                - 1 MB = 1024 * 1024 = 1048576 Bytes
                - 1 GB = 1024 * 1024 * 1024 Bytes
              
              常用示例：
                - 5MB：5 * 1024 * 1024
                - 20MB：20 * 1024 * 1024
                - 100MB：100 * 1024 * 1024
        
        Returns:
            Tuple[bool, str]: 
              返回元组包含：
              - 第1个元素：是否有效（True/False）
              - 第2个元素：错误提示或成功提示
              
              返回示例：
                (True, "✓ 文件大小合法: 2.5 MB")
                (False, "✗ 文件超大: 25.0 MB > 10.0 MB")
        
        异常场景（已处理）：
          - file_obj为None：返回False
          - 文件对象无法读取：返回False
          - 大小超限：返回False，显示实际/限制大小
        
        示例:
            >>> # FastAPI的UploadFile
            >>> from fastapi import UploadFile
            >>> file = UploadFile(file=...)
            >>> is_valid, msg = FileUtils.validate_file_size(file)
            
            >>> # 自定义限制大小（5MB）
            >>> is_valid, msg = FileUtils.validate_file_size(
            ...     file,
            ...     max_size=5*1024*1024
            ... )
            
            >>> # 检查结果
            >>> if is_valid:
            ...     print(f"文件大小有效: {msg}")
            ... else:
            ...     print(f"文件太大: {msg}")
        """
        try:
            # ==================== 步骤1：参数验证 ====================
            # 检查file_obj是否为空
            if file_obj is None:
                return (False, "✗ 文件对象为空")
            
            # 使用默认大小限制（如果未指定）
            if max_size is None:
                max_size = FileUtils.DEFAULT_MAX_FILE_SIZE
            
            # ==================== 步骤2：获取文件大小 ====================
            # 尝试多种方式获取文件大小（兼容不同的文件对象类型）
            file_size = None
            
            # 方法1：FastAPI的UploadFile有.size属性
            if hasattr(file_obj, 'size'):
                file_size = file_obj.size
            
            # 方法2：文件对象的.file属性也有.size
            elif hasattr(file_obj, 'file') and hasattr(file_obj.file, 'size'):
                file_size = file_obj.file.size
            
            # 方法3：使用seek()和tell()方法计算大小（通用方法）
            # 这适用于任何支持seek/tell的类文件对象
            elif hasattr(file_obj, 'seek') and hasattr(file_obj, 'tell'):
                try:
                    # 保存当前位置
                    current_pos = file_obj.tell()
                    
                    # 跳到文件末尾
                    file_obj.seek(0, 2)  # 2=SEEK_END
                    
                    # 获取文件大小
                    file_size = file_obj.tell()
                    
                    # 恢复原位置
                    file_obj.seek(current_pos)
                except:
                    file_size = None
            
            # 方法4：如果是Python文件对象，使用os.fstat
            elif hasattr(file_obj, 'fileno'):
                try:
                    file_size = os.fstat(file_obj.fileno()).st_size
                except:
                    file_size = None
            
            # ==================== 步骤3：处理无法获取大小的情况 ====================
            if file_size is None:
                return (False, "✗ 无法获取文件大小，请检查文件")
            
            # ==================== 步骤4：大小限制检查 ====================
            if file_size > max_size:
                # 转换为更易读的单位（MB）
                file_size_mb = file_size / (1024 * 1024)
                max_size_mb = max_size / (1024 * 1024)
                
                return (
                    False,
                    f"✗ 文件超大: {file_size_mb:.1f} MB > {max_size_mb:.1f} MB"
                )
            
            # ==================== 步骤5：大小验证成功 ====================
            # 转换为更易读的单位（MB）
            file_size_mb = file_size / (1024 * 1024)
            
            return (True, f"✓ 文件大小合法: {file_size_mb:.1f} MB")
            
        except Exception as e:
            # 捕获所有异常
            return (False, f"✗ 文件大小验证失败: {str(e)}")
    
    # ==================== 唯一文件名生成 ====================
    
    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """
        生成唯一的文件名，避免文件覆盖
        
        功能说明：
          - 使用UUID生成随机的唯一标识符
          - 保留原始文件的扩展名
          - 返回新的唯一文件名
        
        实现细节：
          - UUID v4：128位随机数，碰撞概率极低
          - 格式：img_{uuid4}.{原扩展名}
          - 示例：img_abc123def456.png
          - 扩展名大小写：保持原样或转为小写
        
        Args:
            original_filename (str): 
              原始文件名，用于获取扩展名
              示例：'photo.jpg'、'image.PNG'、'document.txt'
        
        Returns:
            str: 
              生成的唯一文件名
              格式：img_{uuid4}.{扩展名}
              示例：'img_550e8400e29b41d4a716446655440000.png'
              
              如果original_filename无扩展名：
                返回：'img_{uuid4}'
        
        异常场景（已处理）：
          - original_filename为空：返回f'img_{uuid}'
          - 无扩展名：返回f'img_{uuid}'
          - 扩展名为空：返回f'img_{uuid}'
        
        示例:
            >>> # 标准文件名
            >>> unique_name = FileUtils.generate_unique_filename('photo.jpg')
            >>> print(unique_name)
            # img_abc123def456.jpg
            
            >>> # PNG文件
            >>> unique_name = FileUtils.generate_unique_filename('image.PNG')
            >>> print(unique_name)
            # img_abc123def456.png
            
            >>> # 无扩展名
            >>> unique_name = FileUtils.generate_unique_filename('README')
            >>> print(unique_name)
            # img_abc123def456
        """
        try:
            # ==================== 步骤1：生成UUID ====================
            # 使用uuid.uuid4()生成128位随机UUID
            # hex属性返回32字符的十六进制字符串，无连接符
            unique_id = uuid.uuid4().hex
            
            # ==================== 步骤2：提取原文件扩展名 ====================
            # 分离文件名和扩展名
            _, file_extension = os.path.splitext(original_filename)
            
            # ==================== 步骤3：组合唯一文件名 ====================
            # 格式：img_{uuid}.{扩展名}
            if file_extension:
                # 有扩展名的情况
                # 扩展名转为小写便于统一处理
                unique_filename = f"img_{unique_id}{file_extension.lower()}"
            else:
                # 无扩展名的情况
                unique_filename = f"img_{unique_id}"
            
            print(f"[INFO] 生成唯一文件名: {original_filename} → {unique_filename}")
            
            return unique_filename
            
        except Exception as e:
            # 异常处理：即使出错也返回一个有效的文件名
            unique_id = uuid.uuid4().hex
            print(f"[ERROR] 生成文件名失败: {e}，使用默认格式")
            return f"img_{unique_id}"
    
    # ==================== 目录管理 ====================
    
    @staticmethod
    def create_dir_if_not_exists(dir_path: str) -> bool:
        """
        如果目录不存在，则创建目录（包括所有父目录）
        
        功能说明：
          - 检查目录是否存在
          - 不存在则递归创建（包括父目录）
          - 返回创建是否成功
        
        实现细节：
          - 使用pathlib.Path便于跨平台兼容性
          - mkdir(parents=True)递归创建父目录
          - exist_ok=True避免目录已存在时报错
          - 完整的异常捕获和错误日志
        
        Args:
            dir_path (str): 
              要创建的目录路径
              支持相对路径和绝对路径
              示例：
                - './uploads'
                - './results'
                - 'D:/data/images'
                - '/var/www/uploads'
        
        Returns:
            bool: 
              True表示目录已存在或创建成功
              False表示创建失败（权限不足、磁盘满等）
        
        异常场景（已处理）：
          - 目录路径为空：返回False
          - 权限不足：返回False，打印错误
          - 磁盘满：返回False，打印错误
          - 路径包含无效字符：返回False
        
        示例:
            >>> # 创建uploads目录
            >>> success = FileUtils.create_dir_if_not_exists('./uploads')
            >>> if success:
            ...     print("目录创建成功")
            
            >>> # 递归创建多层目录
            >>> success = FileUtils.create_dir_if_not_exists('./data/images/2026')
            >>> if success:
            ...     print("多层目录创建成功")
            
            >>> # 创建已存在的目录（不会报错）
            >>> success = FileUtils.create_dir_if_not_exists('./uploads')
            >>> if success:
            ...     print("目录已存在或创建成功")
        """
        try:
            # ==================== 步骤1：参数验证 ====================
            if not dir_path or not isinstance(dir_path, str):
                print(f"[ERROR] 目录路径无效: {dir_path}")
                return False
            
            # ==================== 步骤2：转换为Path对象 ====================
            # pathlib.Path支持跨平台操作（Windows/Linux/Mac）
            path = Path(dir_path)
            
            # ==================== 步骤3：检查目录是否已存在 ====================
            if path.exists():
                # 目录已存在
                if path.is_dir():
                    # 确实是目录
                    print(f"[INFO] 目录已存在: {dir_path}")
                    return True
                else:
                    # 路径存在但不是目录（可能是文件）
                    print(f"[ERROR] 路径存在但不是目录: {dir_path}")
                    return False
            
            # ==================== 步骤4：创建目录 ====================
            try:
                # parents=True：递归创建所有父目录
                # exist_ok=True：目录已存在不报错
                path.mkdir(parents=True, exist_ok=True)
                
                print(f"[SUCCESS] 目录创建成功: {dir_path}")
                return True
                
            except PermissionError:
                # 权限不足
                print(f"[ERROR] 权限不足，无法创建目录: {dir_path}")
                return False
            except OSError as os_error:
                # 其他OS错误（磁盘满、路径无效等）
                print(f"[ERROR] 创建目录失败: {os_error}")
                return False
            
        except Exception as e:
            # 捕获所有其他异常
            print(f"[ERROR] 创建目录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== 文件保存 ====================
    
    @staticmethod
    def save_uploaded_file(
        file_obj,
        save_path: str
    ) -> Tuple[bool, str]:
        """
        保存上传的文件到指定路径
        
        功能说明：
          - 自动创建保存目录（如果不存在）
          - 从file_obj读取文件内容
          - 写入到save_path
          - 返回保存状态和详细提示
        
        实现细节：
          - 自动提取目录路径并创建目录
          - 支持多种file_obj类型（UploadFile、文件对象等）
          - 使用binary mode (wb)保存二进制文件
          - 完整的异常处理和错误日志
        
        Args:
            file_obj: 
              上传的文件对象
              支持类型：
                - FastAPI的UploadFile对象
                - 标准Python文件对象
                - 其他具有.file属性的对象
            
            save_path (str): 
              保存文件的完整路径（包括文件名）
              示例：
                - 'uploads/img_abc123.png'
                - './results/segmented_img.png'
                - 'D:/data/output.jpg'
        
        Returns:
            Tuple[bool, str]: 
              返回元组包含：
              - 第1个元素：保存是否成功（True/False）
              - 第2个元素：成功/失败信息
              
              返回示例：
                (True, "✓ 文件保存成功: uploads/img_abc123.png")
                (False, "✗ 保存失败: 权限不足")
        
        异常场景（已处理）：
          - file_obj为None：返回False，"文件对象为空"
          - save_path为空：返回False，"保存路径无效"
          - 目录创建失败：返回False，"目录创建失败"
          - 文件写入失败：返回False，"文件写入失败"
          - 权限不足：返回False，"权限不足"
        
        示例:
            >>> # FastAPI UploadFile
            >>> from fastapi import UploadFile
            >>> file = UploadFile(...)
            >>> success, msg = FileUtils.save_uploaded_file(
            ...     file,
            ...     'uploads/img_uuid.png'
            ... )
            >>> if success:
            ...     print(f"保存成功: {msg}")
            ... else:
            ...     print(f"保存失败: {msg}")
        """
        try:
            # ==================== 步骤1：参数验证 ====================
            if file_obj is None:
                return (False, "✗ 文件对象为空")
            
            if not save_path or not isinstance(save_path, str):
                return (False, "✗ 保存路径无效")
            
            # ==================== 步骤2：提取目录路径 ====================
            # 从完整路径中分离出目录部分
            save_dir = os.path.dirname(save_path)
            
            # 如果保存路径不含目录（如'file.txt'），使用当前目录
            if not save_dir:
                save_dir = '.'
            
            # ==================== 步骤3：自动创建保存目录 ====================
            if save_dir and not os.path.exists(save_dir):
                success = FileUtils.create_dir_if_not_exists(save_dir)
                if not success:
                    return (False, f"✗ 无法创建目录: {save_dir}")
            
            # ==================== 步骤4：读取file_obj中的内容 ====================
            try:
                # 尝试多种方式读取文件内容
                file_content = None
                
                # 方法1：FastAPI UploadFile有.file属性
                if hasattr(file_obj, 'file'):
                    file_content = file_obj.file.read()
                
                # 方法2：直接读取（标准文件对象）
                elif hasattr(file_obj, 'read'):
                    file_content = file_obj.read()
                
                else:
                    return (False, "✗ 无法读取文件内容")
                
                # ==================== 步骤5：检查文件内容是否为空 ====================
                if not file_content:
                    return (False, "✗ 文件内容为空")
                
            except Exception as read_error:
                return (False, f"✗ 读取文件失败: {str(read_error)}")
            
            # ==================== 步骤6：写入文件到磁盘 ====================
            try:
                # 以二进制写模式打开文件
                with open(save_path, 'wb') as f:
                    # 写入文件内容
                    f.write(file_content)
                
                print(f"[SUCCESS] 文件保存成功: {save_path}")
                
                # ==================== 步骤7：返回成功信息 ====================
                return (True, f"✓ 文件保存成功: {save_path}")
                
            except PermissionError:
                # 权限不足
                return (False, f"✗ 权限不足，无法保存文件: {save_path}")
            except OSError as os_error:
                # 其他OS错误
                return (False, f"✗ 保存失败: {os_error}")
            
        except Exception as e:
            # 捕获所有其他异常
            print(f"[ERROR] 保存文件时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return (False, f"✗ 保存文件失败: {str(e)}")


# ==================== 测试代码（可选） ====================

if __name__ == '__main__':
    """
    FileUtils单元测试示例
    
    运行此文件进行基本功能测试：
    python -m utils.file_utils
    """
    
    print("=" * 60)
    print("FileUtils 单元测试")
    print("=" * 60)
    
    # 测试1：验证文件格式
    print("\n[测试1] 验证文件格式...")
    
    # 有效格式
    is_valid, msg = FileUtils.validate_file_format('image.png')
    print(f"✓ image.png: {msg}")
    assert is_valid
    
    # 大小写不敏感
    is_valid, msg = FileUtils.validate_file_format('image.JPG')
    print(f"✓ image.JPG: {msg}")
    assert is_valid
    
    # 无效格式
    is_valid, msg = FileUtils.validate_file_format('image.gif')
    print(f"✓ image.gif: {msg}")
    assert not is_valid
    
    # 无扩展名
    is_valid, msg = FileUtils.validate_file_format('image')
    print(f"✓ image: {msg}")
    assert not is_valid
    
    # 测试2：生成唯一文件名
    print("\n[测试2] 生成唯一文件名...")
    
    unique_name1 = FileUtils.generate_unique_filename('photo.jpg')
    print(f"✓ 生成1: {unique_name1}")
    assert unique_name1.startswith('img_')
    assert unique_name1.endswith('.jpg')
    
    unique_name2 = FileUtils.generate_unique_filename('image.PNG')
    print(f"✓ 生成2: {unique_name2}")
    assert unique_name2.startswith('img_')
    assert unique_name2.endswith('.png')
    
    # 两次生成的文件名应该不同
    assert unique_name1 != unique_name2
    print(f"✓ 唯一性验证通过")
    
    # 测试3：创建目录
    print("\n[测试3] 创建目录...")
    
    test_dir = './test_uploads'
    success = FileUtils.create_dir_if_not_exists(test_dir)
    assert success
    print(f"✓ 目录创建成功: {test_dir}")
    
    # 再次创建同一目录（应该返回True）
    success = FileUtils.create_dir_if_not_exists(test_dir)
    assert success
    print(f"✓ 目录已存在检查通过")
    
    # 清理测试目录
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"✓ 测试目录已清理")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
