"""
OCTA图像分割平台 - 文件管理数据访问对象（File Data Access Object）

本模块实现文件管理的数据库CRUD操作，负责图片和数据集文件的元信息存储。

功能特点：
  1. 文件元信息管理：记录文件名、路径、类型、大小等信息
  2. 关联模型追踪：支持记录文件关联的模型权重路径
  3. 灵活查询：支持按文件类型筛选（image/dataset）
  4. 完整性保障：数据库记录和本地文件同步删除
  5. 异常处理：完善的错误捕获和友好提示

数据库表结构（file_management表）：
  - id (INTEGER PRIMARY KEY): 记录ID（自增）
  - file_name (TEXT NOT NULL): 原始文件名
  - file_path (TEXT NOT NULL): 本地存储路径
  - file_type (TEXT NOT NULL): 文件类型（'image'、'dataset' 或 'weight'）
  - upload_time (TIMESTAMP): 上传时间（默认当前时间）
  - related_model (TEXT): 关联模型权重路径（可选）
  - file_size (REAL): 文件大小，单位MB（可选）
  - model_type (TEXT): 模型类型（'unet' 或 'rs_unet3_plus'，仅weight类型文件需要）

使用示例：
    >>> from dao.file_dao import FileDAO
    
    >>> # 1. 初始化数据库表
    >>> FileDAO.create_file_table()
    
    >>> # 2. 添加文件记录
    >>> file_id = FileDAO.add_file_record(
    ...     file_name='train_data.zip',
    ...     file_path='uploads/datasets/train_data.zip',
    ...     file_type='dataset',
    ...     file_size=45.6
    ... )
    >>> print(f"文件记录ID: {file_id}")
    
    >>> # 3. 查询所有数据集文件
    >>> datasets = FileDAO.get_file_list(file_type='dataset')
    >>> for file in datasets:
    ...     print(f"{file['file_name']} - {file['file_size']} MB")
    
    >>> # 4. 查询单个文件信息
    >>> file_info = FileDAO.get_file_by_id(1)
    >>> if file_info:
    ...     print(f"文件路径: {file_info['file_path']}")
    
    >>> # 5. 更新文件关联的模型
    >>> success = FileDAO.update_file_relation(1, 'models/weights/unet_v2.pth')
    >>> if success:
    ...     print("模型关联更新成功")
    
    >>> # 6. 删除文件（数据库记录+本地文件）
    >>> success = FileDAO.delete_file(1)
    >>> if success:
    ...     print("文件删除成功")

作者：OCTA Web项目组
日期：2026年1月16日
"""

import sqlite3
import os
import shutil
from datetime import datetime
from typing import Optional, List, Dict

# 导入数据库配置
from config.config import DB_PATH


class FileDAO:
    """
    文件数据访问对象（File Data Access Object）
    
    负责所有与file_management表相关的数据库操作。使用静态方法设计，
    无需实例化。提供文件元信息的完整CRUD功能。
    
    核心职责：
      1. 表管理：创建和维护file_management表结构
      2. 记录增删改查：提供标准CRUD接口
      3. 文件同步：删除数据库记录时同步删除本地文件
      4. 关联追踪：维护文件与模型权重的关联关系
      5. 异常处理：完善的错误捕获和日志记录
    """
    
    # ==================== 表创建 ====================
    
    @staticmethod
    def create_file_table() -> bool:
        """
        创建file_management表（若不存在）
        
        本函数在应用启动时调用，确保数据库表结构存在。使用IF NOT EXISTS
        子句避免重复创建错误。
        
        表字段说明：
          - id: 主键，自增，唯一标识文件记录
          - file_name: 原始文件名（含扩展名）
          - file_path: 文件在服务器上的存储路径（相对或绝对）
          - file_type: 文件类型标识（'image'单张图片、'dataset'数据集压缩包 或 'weight'模型权重）
          - upload_time: 记录创建时间，自动填充当前时间戳
          - related_model: 使用该文件训练/测试的模型权重路径（可选）
          - file_size: 文件大小（MB），用于存储空间统计（可选）
          - model_type: 模型类型（'unet' 或 'rs_unet3_plus'），仅weight类型文件需要，其他类型可为NULL
        
        Returns:
            bool: 创建成功返回True，失败返回False
        
        异常处理：
            捕获所有sqlite3异常，打印错误信息并返回False
        
        示例：
            >>> if FileDAO.create_file_table():
            ...     print("数据库表初始化成功")
            ... else:
            ...     print("数据库表初始化失败")
        """
        try:
            # ========== 步骤1：连接数据库 ==========
            # 使用配置文件中的数据库路径，确保配置统一管理
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # ========== 步骤2：创建表SQL语句 ==========
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS file_management (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                related_model TEXT,
                file_size REAL,
                model_type TEXT
            )
            """
            
            # ========== 步骤3：执行创建表操作 ==========
            cursor.execute(create_table_sql)
            conn.commit()
            
            print("[SUCCESS] file_management表创建成功（或已存在）")
            print(f"[INFO] 数据库路径: {os.path.abspath(DB_PATH)}")
            
            return True
            
        except sqlite3.Error as e:
            # ========== 异常处理：数据库错误 ==========
            print(f"[ERROR] 创建file_management表失败: {e}")
            print(f"[ERROR] 数据库路径: {DB_PATH}")
            return False
            
        finally:
            # ========== 步骤4：关闭数据库连接 ==========
            # 无论成功失败都要关闭连接，避免资源泄露
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    # ==================== 插入记录 ====================
    
    @staticmethod
    def add_file_record(
        file_name: str,
        file_path: str,
        file_type: str,
        related_model: Optional[str] = None,
        file_size: Optional[float] = None,
        model_type: Optional[str] = None
    ) -> Optional[int]:
        """
        添加文件记录到数据库
        
        本函数在文件上传成功后调用，将文件元信息存储到数据库。
        
        Args:
            file_name (str): 原始文件名，如 'train_data.zip' 或 'image_001.png'
            file_path (str): 文件存储路径，如 'uploads/datasets/train_data.zip'
            file_type (str): 文件类型，必须为 'image'、'dataset' 或 'weight'
            related_model (Optional[str]): 关联的模型权重路径，默认None
            file_size (Optional[float]): 文件大小（MB），默认None
            model_type (Optional[str]): 模型类型（'unet' 或 'rs_unet3_plus'），仅weight文件需要，默认None
        
        Returns:
            Optional[int]: 成功返回新插入记录的ID，失败返回None
        
        异常处理：
            - 捕获数据库插入异常
            - 打印详细错误信息（包括参数值）
        
        示例：
            >>> # 添加图片文件
            >>> file_id = FileDAO.add_file_record(
            ...     file_name='octa_001.png',
            ...     file_path='uploads/images/octa_001.png',
            ...     file_type='image',
            ...     file_size=2.5
            ... )
            >>> print(f"文件ID: {file_id}")
            
            >>> # 添加数据集文件
            >>> dataset_id = FileDAO.add_file_record(
            ...     file_name='training_set.zip',
            ...     file_path='uploads/datasets/training_set.zip',
            ...     file_type='dataset',
            ...     related_model='models/weights/unet_octa.pth',
            ...     file_size=120.8
            ... )
        """
        try:
            # ========== 步骤1：参数验证 ==========
            # 验证必填字段非空
            if not file_name or not file_path or not file_type:
                print("[ERROR] 文件名、文件路径、文件类型不能为空")
                return None
            
            # 验证文件类型合法性
            if file_type not in ['image', 'dataset', 'weight']:
                print(f"[ERROR] 文件类型必须为'image'、'dataset'或'weight'，当前值: {file_type}")
                return None
            
            # 验证权重文件必须指定model_type
            if file_type == 'weight' and not model_type:
                print(f"[ERROR] 权重文件必须指定model_type（'unet'或'rs_unet3_plus'）")
                return None
            
            # 验证model_type合法性（如果提供）
            if model_type and model_type not in ['unet', 'rs_unet3_plus']:
                print(f"[ERROR] model_type必须为'unet'或'rs_unet3_plus'，当前值: {model_type}")
                return None
            
            # ========== 步骤2：连接数据库 ==========
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # ========== 步骤3：插入记录SQL语句 ==========
            # upload_time字段使用数据库默认值CURRENT_TIMESTAMP，无需手动指定
            insert_sql = """
            INSERT INTO file_management 
            (file_name, file_path, file_type, related_model, file_size, model_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            # ========== 步骤4：执行插入操作 ==========
            cursor.execute(insert_sql, (file_name, file_path, file_type, related_model, file_size, model_type))
            conn.commit()
            
            # ========== 步骤5：获取新插入记录的ID ==========
            file_id = cursor.lastrowid
            
            print(f"[SUCCESS] 文件记录添加成功")
            print(f"[INFO] 记录ID: {file_id}")
            print(f"[INFO] 文件名: {file_name}")
            print(f"[INFO] 文件类型: {file_type}")
            if file_size:
                print(f"[INFO] 文件大小: {file_size} MB")
            
            return file_id
            
        except sqlite3.Error as e:
            # ========== 异常处理：数据库错误 ==========
            print(f"[ERROR] 添加文件记录失败: {e}")
            print(f"[ERROR] 文件名: {file_name}")
            print(f"[ERROR] 文件路径: {file_path}")
            print(f"[ERROR] 文件类型: {file_type}")
            return None
            
        finally:
            # ========== 步骤6：关闭数据库连接 ==========
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    # ==================== 查询记录 ====================
    
    @staticmethod
    def get_file_list(file_type: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict]:
        """
        查询文件列表，支持按类型筛选
        
        本函数用于文件管理界面展示文件列表，支持查询所有文件或按类型筛选。
        
        Args:
            file_type (Optional[str]): 文件类型筛选，可选值：
                - None: 查询所有文件
                - 'image': 只查询图片文件
                - 'dataset': 只查询数据集文件
                - 'weight': 只查询权重文件
            model_type (Optional[str]): 模型类型筛选（仅当file_type='weight'时有效），可选值：
                - None: 查询所有权重文件
                - 'unet': 只查询U-Net权重

                - 'rs_unet3_plus': 只查询RS-Unet3+权重
        
        Returns:
            List[Dict]: 文件记录列表，每条记录为字典格式，包含所有字段：
                {
                    'id': 1,
                    'file_name': 'train_data.zip',
                    'file_path': 'uploads/datasets/train_data.zip',
                    'file_type': 'dataset',
                    'upload_time': '2026-01-16 10:30:00',
                    'related_model': 'models/weights/unet_octa.pth',
                    'file_size': 45.6
                }
            失败返回空列表 []
        
        异常处理：
            - 捕获数据库查询异常
            - 返回空列表而非抛出异常
        
        示例：
            >>> # 查询所有文件
            >>> all_files = FileDAO.get_file_list()
            >>> print(f"共有{len(all_files)}个文件")
            
            >>> # 查询所有图片
            >>> images = FileDAO.get_file_list(file_type='image')
            >>> for img in images:
            ...     print(f"{img['file_name']}: {img['file_size']} MB")
            
            >>> # 查询所有数据集
            >>> datasets = FileDAO.get_file_list(file_type='dataset')
            >>> for ds in datasets:
            ...     print(f"{ds['file_name']} -> 模型: {ds['related_model']}")
        """
        try:
            # ========== 步骤1：连接数据库 ==========
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row  # 设置row_factory，使查询结果支持字典访问
            cursor = conn.cursor()
            
            # ========== 步骤2：构建查询SQL语句 ==========
            # 使用参数化查询防止SQL注入
            params = []
            
            if file_type and model_type:
                # 同时按文件类型和模型类型筛选（权重文件专用）
                if file_type not in ['image', 'dataset', 'weight']:
                    print(f"[WARNING] 无效的文件类型: {file_type}，仅支持'image'、'dataset'或'weight'")
                    return []
                if model_type not in ['unet', 'rs_unet3_plus']:
                    print(f"[WARNING] 无效的模型类型: {model_type}，仅支持'unet'或'rs_unet3_plus'")
                    return []
                
                query_sql = """
                SELECT id, file_name, file_path, file_type, 
                       upload_time, related_model, file_size, model_type
                FROM file_management
                WHERE file_type = ? AND model_type = ?
                ORDER BY upload_time DESC
                """
                params = [file_type, model_type]
                cursor.execute(query_sql, params)
                print(f"[INFO] 查询文件类型: {file_type}, 模型类型: {model_type}")
                
            elif file_type:
                # 仅按文件类型筛选
                if file_type not in ['image', 'dataset', 'weight']:
                    print(f"[WARNING] 无效的文件类型: {file_type}，仅支持'image'、'dataset'或'weight'")
                    return []
                
                query_sql = """
                SELECT id, file_name, file_path, file_type, 
                       upload_time, related_model, file_size, model_type
                FROM file_management
                WHERE file_type = ?
                ORDER BY upload_time DESC
                """
                params = [file_type]
                cursor.execute(query_sql, params)
                print(f"[INFO] 查询文件类型: {file_type}")
                
            else:
                # 查询所有文件
                query_sql = """
                SELECT id, file_name, file_path, file_type, 
                       upload_time, related_model, file_size, model_type
                FROM file_management
                ORDER BY upload_time DESC
                """
                cursor.execute(query_sql)
                print(f"[INFO] 查询所有文件")
            
            # ========== 步骤3：获取查询结果 ==========
            rows = cursor.fetchall()
            
            # ========== 步骤4：转换为字典列表 ==========
            # sqlite3.Row对象支持dict()转换，便于JSON序列化
            file_list = [dict(row) for row in rows]
            
            print(f"[SUCCESS] 查询成功，找到{len(file_list)}条记录")
            
            return file_list
            
        except sqlite3.Error as e:
            # ========== 异常处理：数据库错误 ==========
            print(f"[ERROR] 查询文件列表失败: {e}")
            return []
            
        finally:
            # ========== 步骤5：关闭数据库连接 ==========
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    @staticmethod
    def get_file_by_id(file_id: int) -> Optional[Dict]:
        """
        按ID查询单个文件信息
        
        本函数用于获取特定文件的详细信息，如编辑页面、详情页面等场景。
        
        Args:
            file_id (int): 文件记录ID
        
        Returns:
            Optional[Dict]: 找到记录返回字典格式，未找到返回None
                字典格式示例：
                {
                    'id': 1,
                    'file_name': 'train_data.zip',
                    'file_path': 'uploads/datasets/train_data.zip',
                    'file_type': 'dataset',
                    'upload_time': '2026-01-16 10:30:00',
                    'related_model': 'models/weights/unet_octa.pth',
                    'file_size': 45.6
                }
        
        异常处理：
            - 捕获数据库查询异常
            - 返回None而非抛出异常
        
        示例：
            >>> # 查询ID为1的文件
            >>> file_info = FileDAO.get_file_by_id(1)
            >>> if file_info:
            ...     print(f"文件名: {file_info['file_name']}")
            ...     print(f"文件大小: {file_info['file_size']} MB")
            ...     print(f"上传时间: {file_info['upload_time']}")
            ... else:
            ...     print("文件不存在")
        """
        try:
            # ========== 步骤1：参数验证 ==========
            if not isinstance(file_id, int) or file_id <= 0:
                print(f"[ERROR] 无效的文件ID: {file_id}")
                return None
            
            # ========== 步骤2：连接数据库 ==========
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row  # 支持字典访问
            cursor = conn.cursor()
            
            # ========== 步骤3：查询SQL语句 ==========
            query_sql = """
            SELECT id, file_name, file_path, file_type, 
                   upload_time, related_model, file_size
            FROM file_management
            WHERE id = ?
            """
            
            # ========== 步骤4：执行查询 ==========
            cursor.execute(query_sql, (file_id,))
            row = cursor.fetchone()
            
            # ========== 步骤5：处理查询结果 ==========
            if row:
                file_info = dict(row)
                print(f"[SUCCESS] 查询成功，文件ID: {file_id}")
                print(f"[INFO] 文件名: {file_info['file_name']}")
                return file_info
            else:
                print(f"[WARNING] 文件不存在，ID: {file_id}")
                return None
            
        except sqlite3.Error as e:
            # ========== 异常处理：数据库错误 ==========
            print(f"[ERROR] 查询文件失败: {e}")
            print(f"[ERROR] 文件ID: {file_id}")
            return None
            
        finally:
            # ========== 步骤6：关闭数据库连接 ==========
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    # ==================== 删除记录 ====================
    
    @staticmethod
    def delete_file(file_id: int) -> bool:
        """
        删除文件记录和本地文件
        
        本函数执行完整的文件删除操作：
        1. 从数据库查询文件路径
        2. 删除数据库记录
        3. 删除本地文件或文件夹
        
        关键特性：
          - 数据库与文件系统同步：确保记录删除后文件也被删除
          - 文件/目录区分：自动判断是文件还是目录，使用对应删除方法
          - 路径校验：删除前检查路径存在性，避免报错
          - 原子性：先删除数据库记录，再删除文件（防止记录残留）
        
        Args:
            file_id (int): 要删除的文件记录ID
        
        Returns:
            bool: 删除成功返回True，失败返回False
        
        异常处理：
            - 捕获数据库删除异常
            - 捕获文件系统删除异常
            - 详细错误日志
        
        示例：
            >>> # 删除单个图片文件
            >>> if FileDAO.delete_file(1):
            ...     print("文件删除成功")
            ... else:
            ...     print("文件删除失败")
            
            >>> # 删除数据集目录
            >>> if FileDAO.delete_file(2):
            ...     print("数据集目录删除成功")
        """
        try:
            # ========== 步骤1：查询文件路径 ==========
            # 先获取文件信息，确保文件存在并获取路径
            file_info = FileDAO.get_file_by_id(file_id)
            
            if not file_info:
                print(f"[ERROR] 文件记录不存在，ID: {file_id}")
                return False
            
            file_path = file_info['file_path']
            file_name = file_info['file_name']
            
            # ========== 步骤2：删除数据库记录 ==========
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            delete_sql = "DELETE FROM file_management WHERE id = ?"
            cursor.execute(delete_sql, (file_id,))
            conn.commit()
            
            print(f"[SUCCESS] 数据库记录删除成功，ID: {file_id}")
            
            # ========== 步骤3：删除本地文件或目录 ==========
            # 检查路径存在性，避免FileNotFoundError
            if os.path.exists(file_path):
                # 判断是文件还是目录
                if os.path.isfile(file_path):
                    # 删除文件
                    os.remove(file_path)
                    print(f"[SUCCESS] 本地文件删除成功: {file_path}")
                elif os.path.isdir(file_path):
                    # 删除目录及其所有内容
                    shutil.rmtree(file_path)
                    print(f"[SUCCESS] 本地目录删除成功: {file_path}")
                else:
                    print(f"[WARNING] 路径类型未知: {file_path}")
            else:
                print(f"[WARNING] 本地文件不存在（可能已被手动删除）: {file_path}")
            
            print(f"[SUCCESS] 文件'{file_name}'删除完成")
            return True
            
        except sqlite3.Error as e:
            # ========== 异常处理：数据库错误 ==========
            print(f"[ERROR] 数据库删除失败: {e}")
            print(f"[ERROR] 文件ID: {file_id}")
            return False
            
        except OSError as e:
            # ========== 异常处理：文件系统错误 ==========
            print(f"[ERROR] 本地文件删除失败: {e}")
            print(f"[ERROR] 文件路径: {file_path}")
            # 数据库记录已删除，但文件删除失败
            # 可以考虑回滚，但通常文件删除失败不影响业务
            return False
            
        finally:
            # ========== 步骤4：关闭数据库连接 ==========
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    # ==================== 更新记录 ====================
    
    @staticmethod
    def update_file_relation(file_id: int, related_model: str) -> bool:
        """
        更新文件关联的模型权重
        
        本函数用于在模型训练完成后，将训练使用的文件（数据集）与
        生成的模型权重文件建立关联关系，便于追踪模型来源。
        
        Args:
            file_id (int): 文件记录ID
            related_model (str): 模型权重文件路径，如 'models/weights/unet_v2.pth'
        
        Returns:
            bool: 更新成功返回True，失败返回False
        
        异常处理：
            - 捕获数据库更新异常
            - 打印详细错误信息
        
        示例：
            >>> # 训练完成后关联模型权重
            >>> success = FileDAO.update_file_relation(
            ...     file_id=1,
            ...     related_model='models/weights/unet_trained_20260116.pth'
            ... )
            >>> if success:
            ...     print("模型关联更新成功")
            
            >>> # 批量更新多个文件的关联
            >>> file_ids = [1, 2, 3]
            >>> model_path = 'models/weights/unet_best.pth'
            >>> for fid in file_ids:
            ...     FileDAO.update_file_relation(fid, model_path)
        """
        try:
            # ========== 步骤1：参数验证 ==========
            if not isinstance(file_id, int) or file_id <= 0:
                print(f"[ERROR] 无效的文件ID: {file_id}")
                return False
            
            if not related_model:
                print(f"[ERROR] 模型路径不能为空")
                return False
            
            # ========== 步骤2：连接数据库 ==========
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # ========== 步骤3：更新SQL语句 ==========
            update_sql = """
            UPDATE file_management
            SET related_model = ?
            WHERE id = ?
            """
            
            # ========== 步骤4：执行更新操作 ==========
            cursor.execute(update_sql, (related_model, file_id))
            conn.commit()
            
            # ========== 步骤5：检查更新结果 ==========
            # cursor.rowcount表示受影响的行数
            if cursor.rowcount > 0:
                print(f"[SUCCESS] 模型关联更新成功")
                print(f"[INFO] 文件ID: {file_id}")
                print(f"[INFO] 关联模型: {related_model}")
                return True
            else:
                print(f"[WARNING] 文件不存在，ID: {file_id}")
                return False
            
        except sqlite3.Error as e:
            # ========== 异常处理：数据库错误 ==========
            print(f"[ERROR] 更新模型关联失败: {e}")
            print(f"[ERROR] 文件ID: {file_id}")
            print(f"[ERROR] 模型路径: {related_model}")
            return False
            
        finally:
            # ========== 步骤6：关闭数据库连接 ==========
            if 'conn' in locals():
                cursor.close()
                conn.close()


# ==================== 模块初始化 ====================

# 在模块导入时自动创建数据库表
# 确保表结构存在，避免后续操作失败
print("[INFO] 初始化文件管理数据库...")
if FileDAO.create_file_table():
    print("[INFO] 文件管理模块加载成功")
else:
    print("[WARNING] 文件管理模块加载失败，请检查数据库配置")
