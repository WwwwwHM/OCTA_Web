"""OCTA图像分割平台 - 数据访问对象（Image Data Access Object）

本模块实现了ImageDAO类，专门负责OCTA图像分割数据的SQLite数据库操作。

架构设计理念：
  1. 数据库隔离：所有SQL操作封装在DAO类中，不散布于业务逻辑
  2. 接口清晰：提供明确的CRUD接口，隐藏SQL细节
  3. 异常处理：所有数据库异常都被捕获和记录
  4. 资源管理：确保连接和游标的正确关闭，避免资源泄露
  5. 扩展性：易于添加新字段或修改表结构

数据库表结构（images表）：
  - id (INTEGER PRIMARY KEY): 记录ID
  - filename (TEXT UNIQUE): 原始文件名
  - upload_time (TEXT): 上传时间（ISO 8601格式）
  - model_type (TEXT): 使用的模型类型（'unet' 或 'fcn'）
  - original_path (TEXT): 原始图像文件路径
  - result_path (TEXT): 分割结果文件路径

作者：OCTA Web项目组
日期：2026年1月14日
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

# 导入配置（所有常量来自config.py，确保配置集中管理）
from config import (
    DB_PATH,          # 数据库文件路径
    DB_TABLE_NAME     # 数据库表名
)


class ImageDAO:
    """
    图像数据访问对象（Data Access Object）
    
    负责所有与images表相关的数据库操作。使用静态方法/类方法设计，
    无需实例化。支持多数据库实例（通过参数传递db_path）。
    
    核心特点：
      1. 所有方法接受db_path参数（默认"./octa.db"）
      2. 自动关闭游标和连接，避免资源泄露
      3. 完整的异常处理和错误日志
      4. 返回标准化的数据结构（List[Dict]）
      5. SQL语句集中管理，便于维护
    
    使用示例：
        >>> # 初始化数据库
        >>> ImageDAO.init_db('./octa.db')
        
        >>> # 插入记录
        >>> record_id = ImageDAO.insert_record(
        ...     filename='img_abc123.png',
        ...     upload_time='2026-01-14T10:30:00',
        ...     model_type='unet',
        ...     original_path='uploads/img_abc123.png',
        ...     result_path='results/img_abc123_seg.png',
        ...     db_path='./octa.db'
        ... )
        >>> print(f"插入成功，记录ID: {record_id}")
        
        >>> # 查询所有记录
        >>> records = ImageDAO.get_all_records('./octa.db')
        >>> for record in records:
        ...     print(f"ID: {record['id']}, 模型: {record['model_type']}")
        
        >>> # 查询单条记录
        >>> record = ImageDAO.get_record_by_id(1, './octa.db')
        >>> if record:
        ...     print(f"找到记录: {record['filename']}")
        
        >>> # 删除记录
        >>> success = ImageDAO.delete_record_by_id(1, './octa.db')
        >>> if success:
        ...     print("记录删除成功")
    """
    
    # ==================== SQL语句常量 ====================
    # 集中管理所有SQL语句，便于维护和修改
    # 注：表名使用DB_TABLE_NAME配置，支持动态表名
    
    @staticmethod
    def _build_create_table_sql():
        """构建CREATE TABLE语句，使用配置中的表名
        
        【2026.1.20更新】model_type字段用于区分权重所属模型：
          - 'unet': U-Net模型权重
          - 'rs_unet3_plus': RS-Unet3+模型权重
          - 其他值向后兼容
        """
        return f"""
    CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        upload_time TEXT NOT NULL,
        model_type TEXT NOT NULL,
        original_path TEXT NOT NULL,
        result_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT valid_model_type CHECK (model_type IN ('unet', 'rs_unet3_plus', 'fcn'))
    )
    """
    
    # 为了兼容性，保留类变量（在init_db中使用动态版本）
    CREATE_TABLE_SQL = "将由init_db中的动态SQL替代"
    """建表语句：创建数据表（表名来自DB_TABLE_NAME配置）
    
    字段说明：
      - id: 自增主键，唯一标识一条记录
      - filename: 上传文件名，UNIQUE约束保证不重复
      - upload_time: 上传时间（ISO 8601格式，如'2026-01-14T10:30:00'）
      - model_type: 分割模型类型（'unet' 或 'fcn'）
      - original_path: 原始图像存储路径
      - result_path: 分割结果存储路径
      - created_at: 数据库记录创建时间（自动时间戳）
    
    约束：
      - filename是UNIQUE，避免重复上传
      - 所有字段除created_at外都是NOT NULL
    
    扩展建议：
      - 可添加user_id字段用于多用户支持
      - 可添加file_size字段记录文件大小
      - 可添加process_time字段记录处理时间
      - 可添加status字段记录处理状态
    """
    
    # ==================== 数据库初始化 ====================
    
    @staticmethod
    def init_db(db_path: str = None) -> bool:
        """
        初始化数据库，创建images表（若不存在）
        
        功能说明：
          - 检查数据库文件是否存在
          - 创建数据库连接（自动创建文件if not exists）
          - 执行CREATE TABLE IF NOT EXISTS语句
          - 自动提交和关闭连接
        
        处理流程：
          1. 获取数据库目录路径
          2. 确保目录存在（create parent directories）
          3. 连接到数据库
          4. 执行建表语句（IF NOT EXISTS）
          5. 提交事务
          6. 关闭连接
        
        Args:
            db_path (str, optional): 
              数据库文件路径，默认为'./octa.db'。
              支持相对路径和绝对路径。
              示例：'./octa.db' 或 'D:/data/octa.db'
        
        Returns:
            bool: 
              True表示初始化成功（表已创建或已存在）
              False表示初始化失败（异常发生）
        
        异常场景（已处理）：
          - 目录不存在：自动创建
          - 数据库文件损坏：异常返回False
          - 权限不足：异常返回False
          - 表已存在：IF NOT EXISTS处理，正常返回True
        
        示例:
            >>> success = ImageDAO.init_db()  # 使用默认配置
            >>> if success:
            ...     print("✓ 数据库初始化成功")
            ... else:
            ...     print("✗ 数据库初始化失败")
        """
        # 使用config的默认路径
        if db_path is None:
            db_path = DB_PATH
        
        try:
            # ==================== 步骤1：检查和创建目录 ====================
            # 获取数据库文件所在的目录
            db_dir = os.path.dirname(db_path)
            
            # 如果目录不为空且不存在，创建所有必要的目录（包括父目录）
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"[INFO] 数据库目录已创建: {db_dir}")
            
            # ==================== 步骤2：连接到数据库 ====================
            # sqlite3.connect()：如果数据库文件不存在会自动创建
            conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # ==================== 步骤3：执行建表语句 ====================
            # IF NOT EXISTS确保重复调用本函数也不会出错
            # 使用动态SQL语句，表名来自DB_TABLE_NAME配置
            create_sql = ImageDAO._build_create_table_sql()
            conn.execute(create_sql)
            
            # ==================== 步骤4：提交事务 ====================
            # 显式提交，确保表创建被保存到磁盘
            conn.commit()
            
            print(f"[SUCCESS] 数据库初始化成功: {db_path}")
            
            # ==================== 步骤5：关闭连接 ====================
            # 及时关闭连接，避免连接泄露
            conn.close()
            
            return True
            
        except sqlite3.Error as db_error:
            # SQLite特定异常
            print(f"[ERROR] 数据库初始化失败（SQLite错误）: {db_error}")
            return False
        except Exception as e:
            # 其他异常（如权限错误、磁盘满等）
            print(f"[ERROR] 数据库初始化失败（系统错误）: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== 插入操作（Create） ====================
    
    @staticmethod
    def insert_record(
        filename: str,
        upload_time: str,
        model_type: str,
        original_path: str,
        result_path: str,
        db_path: str = None
    ) -> Optional[int]:
        """
        插入一条分割记录到数据库
        
        功能说明：
          - 向images表插入一条新记录
          - 自动生成record_id（自增）
          - 返回插入的记录ID或None（失败）
        
        插入流程：
          1. 建立数据库连接
          2. 参数化查询（防SQL注入）
          3. 执行INSERT语句
          4. 提交事务
          5. 获取插入的行ID
          6. 关闭连接
        
        Args:
            filename (str): 
              上传的文件名（应为UUID+扩展名，如'img_abc123def456.png'）
              UNIQUE约束：同一文件名不能重复插入
            
            upload_time (str): 
              上传时间（ISO 8601格式，如'2026-01-14T10:30:00'）
              建议使用datetime.datetime.now().isoformat()生成
            
            model_type (str): 
              使用的模型类型（'unet' 或 'fcn'）
              小写字符串，无需验证
            
            original_path (str): 
              原始图像在服务器上的存储路径（相对或绝对）
              如'uploads/img_abc123def456.png'
            
            result_path (str): 
              分割结果在服务器上的存储路径
              如'results/img_abc123def456_seg.png'
            
            db_path (str, optional): 
              数据库文件路径，默认'./octa.db'
        
        Returns:
            Optional[int]: 
              成功时返回新插入记录的ID（正整数）
              失败时返回None
        
        异常场景（已处理）：
          - 文件名重复（UNIQUE约束冲突）：返回None，打印警告
          - 数据库文件损坏：返回None，打印错误
          - 参数类型错误：返回None，打印错误
          - 数据库连接失败：返回None，打印错误
        
        示例:
            >>> record_id = ImageDAO.insert_record(
            ...     filename='img_uuid_12345.png',
            ...     upload_time='2026-01-14T10:30:00',
            ...     model_type='unet',
            ...     original_path='uploads/img_uuid_12345.png',
            ...     result_path='results/img_uuid_12345_seg.png'
            ... )
            >>> if record_id:
            ...     print(f"✓ 记录插入成功，ID: {record_id}")
            ... else:
            ...     print("✗ 记录插入失败")
        """        # 使用config的默认路径
        if db_path is None:
            db_path = DB_PATH
                # 使用config的默认路径
        if db_path is None:
            db_path = DB_PATH
        
        try:
            # ==================== 步骤1：建立数据库连接 ====================
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # ==================== 步骤2：参数化SQL查询（防SQL注入） ====================
            # 使用?占位符，参数通过元组传递
            # 这是防止SQL注入的标准做法
            # 表名使用DB_TABLE_NAME配置
            sql = f"""
            INSERT INTO {DB_TABLE_NAME} (filename, upload_time, model_type, original_path, result_path)
            VALUES (?, ?, ?, ?, ?)
            """
            
            # ==================== 步骤3：执行INSERT语句 ====================
            cursor.execute(sql, (filename, upload_time, model_type, original_path, result_path))
            
            # ==================== 步骤4：提交事务 ====================
            # 显式提交，确保数据被写入数据库
            conn.commit()
            
            # ==================== 步骤5：获取插入的行ID ====================
            # cursor.lastrowid获取最后插入的自增ID
            record_id = cursor.lastrowid
            
            print(f"[SUCCESS] 记录插入成功（ID={record_id}）: {filename}")
            
            return record_id
            
        except sqlite3.IntegrityError as integrity_error:
            # UNIQUE约束冲突或其他完整性错误
            if 'UNIQUE constraint failed' in str(integrity_error):
                print(f"[WARNING] 文件名重复，插入失败: {filename}")
            else:
                print(f"[ERROR] 数据完整性错误: {integrity_error}")
            return None
        except sqlite3.Error as db_error:
            # SQLite特定异常
            print(f"[ERROR] 数据库操作失败: {db_error}")
            return None
        except Exception as e:
            # 其他异常（如参数类型错误）
            print(f"[ERROR] 插入记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # ==================== 步骤6：关闭连接 ====================
            # 确保游标和连接被关闭，避免资源泄露
            try:
                cursor.close()
                conn.close()
            except:
                pass  # 关闭时出错不影响返回值
    
    # ==================== 查询操作（Read） ====================
    
    @staticmethod
    def get_all_records(db_path: str = None) -> List[Dict]:
        """
        查询所有分割记录，按上传时间倒序排列
        
        功能说明：
          - 从images表查询所有记录
          - 按upload_time倒序排列（最新的在前）
          - 返回列表，每个元素是一条记录的字典
        
        查询流程：
          1. 建立数据库连接
          2. 设置行工厂为sqlite3.Row（支持字典访问）
          3. 执行SELECT语句（ORDER BY upload_time DESC）
          4. 获取所有结果
          5. 转换为列表
          6. 关闭连接
        
        Args:
            db_path (str, optional): 
              数据库文件路径，默认'./octa.db'
        
        Returns:
            List[Dict]: 
              记录列表，按上传时间倒序
              空表时返回[]（空列表）
              
              列表中每个元素是一个字典，包含以下键：
                - 'id' (int): 记录ID
                - 'filename' (str): 文件名
                - 'upload_time' (str): 上传时间
                - 'model_type' (str): 模型类型
                - 'original_path' (str): 原始路径
                - 'result_path' (str): 结果路径
                - 'created_at' (str): 创建时间（数据库自动）
              
              示例：
                [
                  {
                    'id': 2,
                    'filename': 'img_new.png',
                    'upload_time': '2026-01-14T11:00:00',
                    'model_type': 'unet',
                    'original_path': 'uploads/img_new.png',
                    'result_path': 'results/img_new_seg.png',
                    'created_at': '2026-01-14 11:00:00'
                  },
                  {
                    'id': 1,
                    'filename': 'img_old.png',
                    'upload_time': '2026-01-14T10:30:00',
                    'model_type': 'fcn',
                    'original_path': 'uploads/img_old.png',
                    'result_path': 'results/img_old_seg.png',
                    'created_at': '2026-01-14 10:30:00'
                  }
                ]
        
        异常场景（已处理）：
          - 表不存在：返回[]
          - 数据库文件损坏：返回[]
          - 连接失败：返回[]
          - 无记录：返回[]（正常情况）
        
        示例:
            >>> records = ImageDAO.get_all_records()
            >>> if records:
            ...     print(f"✓ 找到 {len(records)} 条记录")
            ...     for record in records:
            ...         print(f"  ID: {record['id']}, 模型: {record['model_type']}")
            ... else:
            ...     print("✗ 数据库中无记录或查询失败")
        """
        # 使用config的默认路径
        if db_path is None:
            db_path = DB_PATH
        
        try:
            # ==================== 步骤1：建立数据库连接 ====================
            conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # ==================== 步骤2：设置行工厂为sqlite3.Row ====================
            # 这样查询结果可以像字典一样访问（按列名），而不是元组
            # 使得返回的数据结构更清晰易用
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # ==================== 步骤3：执行SELECT查询 ====================
            # ORDER BY upload_time DESC：按上传时间倒序
            # 这样最新上传的记录会首先出现
            # 表名使用DB_TABLE_NAME配置
            sql = f"SELECT * FROM {DB_TABLE_NAME} ORDER BY upload_time DESC"
            cursor.execute(sql)
            
            # ==================== 步骤4：获取所有结果 ====================
            rows = cursor.fetchall()
            
            # ==================== 步骤5：转换为列表 ====================
            # 将sqlite3.Row对象转换为字典列表
            records = [dict(row) for row in rows]
            
            print(f"[INFO] 查询成功，找到 {len(records)} 条记录")
            
            return records
            
        except sqlite3.OperationalError as op_error:
            # 表不存在等操作性错误
            if 'no such table' in str(op_error):
                print(f"[WARNING] {DB_TABLE_NAME}表不存在，返回空列表")
            else:
                print(f"[ERROR] 数据库操作错误: {op_error}")
            return []
        except sqlite3.Error as db_error:
            # SQLite特定异常
            print(f"[ERROR] 数据库查询失败: {db_error}")
            return []
        except Exception as e:
            # 其他异常
            print(f"[ERROR] 查询所有记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # ==================== 步骤6：关闭连接 ====================
            try:
                cursor.close()
                conn.close()
            except:
                pass
    
    @staticmethod
    def get_record_by_id(record_id: int, db_path: str = None) -> Optional[Dict]:
        """
        按ID查询单条分割记录
        
        功能说明：
          - 按primary key查询指定ID的记录
          - 返回字典格式或None（未找到）
          - 查询最快（使用主键索引）
        
        查询流程：
          1. 参数验证（ID为正整数）
          2. 建立数据库连接
          3. 设置行工厂为sqlite3.Row
          4. 执行WHERE查询
          5. 获取单行结果
          6. 关闭连接
        
        Args:
            record_id (int): 
              要查询的记录ID（正整数）
              必须是数据库中存在的ID
            
            db_path (str, optional): 
              数据库文件路径，默认'./octa.db'
        
        Returns:
            Optional[Dict]: 
              找到记录时返回字典格式的记录
              未找到或出错时返回None
              
              返回字典包含的字段同get_all_records()
        
        异常场景（已处理）：
          - ID不存在：返回None（正常情况）
          - ID类型错误：返回None
          - 表不存在：返回None
          - 数据库文件损坏：返回None
        
        示例:
            >>> record = ImageDAO.get_record_by_id(1)
            >>> if record:
            ...     print(f"✓ 找到记录: {record['filename']}")
            ...     print(f"  上传时间: {record['upload_time']}")
            ...     print(f"  模型: {record['model_type']}")
            ... else:
            ...     print("✗ 未找到该ID的记录")
        """
        # 使用config的默认路径
        if db_path is None:
            db_path = DB_PATH
        
        try:
            # ==================== 步骤1：参数验证 ====================
            # 检查ID是否为正整数
            if not isinstance(record_id, int) or record_id <= 0:
                print(f"[ERROR] 无效的记录ID: {record_id}，必须为正整数")
                return None
            
            # ==================== 步骤2：建立数据库连接 ====================
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # ==================== 步骤3：执行WHERE查询 ====================
            # 使用参数化查询，ID通过?占位符传递
            # 表名使用DB_TABLE_NAME配置
            sql = f"SELECT * FROM {DB_TABLE_NAME} WHERE id = ?"
            cursor.execute(sql, (record_id,))
            
            # ==================== 步骤4：获取单行结果 ====================
            # fetchone()返回单行或None（未找到）
            row = cursor.fetchone()
            
            # ==================== 步骤5：转换结果 ====================
            if row:
                record = dict(row)
                print(f"[INFO] 找到记录: ID={record_id}, 文件名={record['filename']}")
                return record
            else:
                print(f"[WARNING] 未找到ID为 {record_id} 的记录")
                return None
            
        except sqlite3.OperationalError as op_error:
            if 'no such table' in str(op_error):
                print(f"[WARNING] {DB_TABLE_NAME}表不存在")
            else:
                print(f"[ERROR] 数据库操作错误: {op_error}")
            return None
        except sqlite3.Error as db_error:
            print(f"[ERROR] 数据库查询失败: {db_error}")
            return None
        except Exception as e:
            print(f"[ERROR] 查询记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # ==================== 步骤6：关闭连接 ====================
            try:
                cursor.close()
                conn.close()
            except:
                pass
    
    # ==================== 删除操作（Delete） ====================
    
    @staticmethod
    def delete_record_by_id(record_id: int, db_path: str = None) -> bool:
        """
        按ID删除单条分割记录
        
        功能说明：
          - 根据primary key删除指定ID的记录
          - 返回True表示删除成功，False表示失败或ID不存在
          - 注意：此操作是不可逆的，请谨慎使用
        
        删除流程：
          1. 参数验证（ID为正整数）
          2. 建立数据库连接
          3. 执行DELETE语句
          4. 检查是否有行被删除
          5. 提交事务
          6. 关闭连接
        
        Args:
            record_id (int): 
              要删除的记录ID（正整数）
              必须是数据库中存在的ID
            
            db_path (str, optional): 
              数据库文件路径，默认'./octa.db'
        
        Returns:
            bool: 
              True：删除成功（有至少一行被删除）
              False：删除失败（ID不存在或其他错误）
        
        异常场景（已处理）：
          - ID不存在：返回False（正常情况）
          - ID类型错误：返回False
          - 表不存在：返回False
          - 数据库锁定：返回False
          - 权限不足：返回False
        
        示例:
            >>> success = ImageDAO.delete_record_by_id(1)
            >>> if success:
            ...     print("✓ 记录删除成功")
            ... else:
            ...     print("✗ 删除失败，可能是ID不存在")
        
        注意事项：
            ⚠️ 此操作不可逆，请确保要删除的记录ID正确
            ⚠️ 删除记录后，关联的文件（uploads/results）需手动删除
            ⚠️ 建议在删除前先调用get_record_by_id()确认记录存在
        """
        # 使用config的默认路径
        if db_path is None:
            db_path = DB_PATH
        
        try:
            # ==================== 步骤1：参数验证 ====================
            # 检查ID是否为正整数
            if not isinstance(record_id, int) or record_id <= 0:
                print(f"[ERROR] 无效的记录ID: {record_id}，必须为正整数")
                return False
            
            # ==================== 步骤2：建立数据库连接 ====================
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # ==================== 步骤3：执行DELETE语句 ====================
            # 使用参数化查询，ID通过?占位符传递
            # 表名使用DB_TABLE_NAME配置
            sql = f"DELETE FROM {DB_TABLE_NAME} WHERE id = ?"
            cursor.execute(sql, (record_id,))
            
            # ==================== 步骤4：检查是否有行被删除 ====================
            # cursor.rowcount表示受影响的行数
            rows_deleted = cursor.rowcount
            
            if rows_deleted > 0:
                # ==================== 步骤5：提交事务 ====================
                # 显式提交，确保删除被保存到数据库
                conn.commit()
                print(f"[SUCCESS] 记录删除成功（ID={record_id}）")
                return True
            else:
                # 没有行被删除，说明ID不存在
                print(f"[WARNING] 未找到ID为 {record_id} 的记录，无法删除")
                return False
            
        except sqlite3.OperationalError as op_error:
            if 'no such table' in str(op_error):
                print(f"[WARNING] {DB_TABLE_NAME}表不存在")
            else:
                print(f"[ERROR] 数据库操作错误: {op_error}")
            return False
        except sqlite3.DatabaseError as db_error:
            # 数据库被锁定等错误
            print(f"[ERROR] 数据库锁定或其他错误: {db_error}")
            return False
        except sqlite3.Error as db_error:
            # 其他SQLite错误
            print(f"[ERROR] 数据库删除失败: {db_error}")
            return False
        except Exception as e:
            # 其他异常
            print(f"[ERROR] 删除记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # ==================== 步骤6：关闭连接 ====================
            try:
                cursor.close()
                conn.close()
            except:
                pass


# ==================== 测试代码（可选） ====================

if __name__ == '__main__':
    """
    DAO单元测试示例
    
    运行此文件进行基本功能测试：
    python -m octa_backend.dao.image_dao
    """
    
    import sys
    
    print("=" * 60)
    print("ImageDAO 单元测试")
    print("=" * 60)
    
    test_db = './test_octa.db'
    
    # 清除旧的测试数据库
    if os.path.exists(test_db):
        os.remove(test_db)
        print("[CLEAN] 旧测试数据库已删除")
    
    # 测试1：初始化数据库
    print("\n[测试1] 初始化数据库...")
    success = ImageDAO.init_db(test_db)
    assert success, "✗ 初始化失败"
    print("✓ 初始化成功")
    
    # 测试2：插入记录
    print("\n[测试2] 插入记录...")
    id1 = ImageDAO.insert_record(
        filename='test_img1.png',
        upload_time=datetime.now().isoformat(),
        model_type='unet',
        original_path='uploads/test_img1.png',
        result_path='results/test_img1_seg.png',
        db_path=test_db
    )
    assert id1 is not None, "✗ 插入失败"
    print(f"✓ 插入成功，ID={id1}")
    
    id2 = ImageDAO.insert_record(
        filename='test_img2.png',
        upload_time=datetime.now().isoformat(),
        model_type='fcn',
        original_path='uploads/test_img2.png',
        result_path='results/test_img2_seg.png',
        db_path=test_db
    )
    assert id2 is not None, "✗ 插入失败"
    print(f"✓ 插入成功，ID={id2}")
    
    # 测试3：查询所有记录
    print("\n[测试3] 查询所有记录...")
    records = ImageDAO.get_all_records(test_db)
    assert len(records) == 2, f"✗ 应有2条记录，实际{len(records)}条"
    print(f"✓ 查询成功，找到{len(records)}条记录")
    
    # 测试4：按ID查询
    print("\n[测试4] 按ID查询...")
    record = ImageDAO.get_record_by_id(id1, test_db)
    assert record is not None, "✗ 查询失败"
    assert record['filename'] == 'test_img1.png', "✗ 数据不匹配"
    print(f"✓ 查询成功: {record['filename']}")
    
    # 测试5：删除记录
    print("\n[测试5] 删除记录...")
    success = ImageDAO.delete_record_by_id(id1, test_db)
    assert success, "✗ 删除失败"
    print(f"✓ 删除成功（ID={id1}）")
    
    # 测试6：验证删除结果
    print("\n[测试6] 验证删除结果...")
    records = ImageDAO.get_all_records(test_db)
    assert len(records) == 1, f"✗ 应有1条记录，实际{len(records)}条"
    assert records[0]['filename'] == 'test_img2.png', "✗ 数据不匹配"
    print(f"✓ 验证成功，剩余{len(records)}条记录")
    
    # 清理测试数据库
    os.remove(test_db)
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
