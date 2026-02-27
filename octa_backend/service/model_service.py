"""
OCTA图像分割平台 - AI模型服务层（Model Service Layer）

本模块封装所有与AI模型相关的操作，包括：
- 模型加载（U-Net/RS-Unet3+）
- 图像预处理（尺寸调整、归一化）
- 模型推理（前向传播）
- 结果后处理（转灰度图、保存PNG）

架构设计理念：
  1. 业务逻辑解耦：Controller不直接调用models，而是通过Service层
  2. 模型抽象：屏蔽底层模型细节，提供统一的分割接口
  3. 易于扩展：添加新模型只需在Service层新增方法
  4. 错误隔离：模型相关错误在Service层处理，不传播到Controller
  5. 可测试性：Service层可独立进行单元测试

AI模型处理流程：
  1. 加载模型（load_model）
  2. 预处理图像（preprocess_image）→ 256x256 Tensor
  3. 模型推理（segment_image内部）→ 分割掩码
  4. 后处理结果（postprocess_result）→ PNG文件
  5. 返回结果路径给Controller

作者：OCTA Web项目组
日期：2026年1月14日
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple

# 导入配置（模型参数来自配置文件，便于适配不同OCTA数据集）
from config.config import (
    UNET_WEIGHT_PATH,      # U-Net预训练权重路径
    RS_UNET3_PLUS_WEIGHT_DIR,  # RS-Unet3+权重目录（默认查找best权重）
    IMAGE_TARGET_SIZE,     # 图像预处理目标尺寸
    MODEL_DEVICE,          # 模型运行设备
    DEFAULT_MODEL_TYPE     # 默认模型类型
)

# 导入 RS-Unet3+ 模型
from models.rs_unet3_plus import RSUNet3Plus


# ==================== U-Net模型定义 ====================

class DoubleConv(nn.Module):
    """
    U-Net基础卷积块（Double Convolution Block）
    
    结构：Conv2d → BN → ReLU → Conv2d → BN → ReLU
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """
    U-Net分割模型（256x256输入）
    
    网络架构：
    - 编码器：256→128→64→32→16（4次下采样）
    - 瓶颈层：16x16, 1024通道
    - 解码器：16→32→64→128→256（4次上采样）
    - 跳跃连接：保留细节信息
    
    输入：(batch_size, 3, 256, 256) - RGB图像
    输出：(batch_size, 1, 256, 256) - 分割掩码
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(UNet, self).__init__()
        
        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器（注意跳跃连接后的通道数）
        self.dec1 = DoubleConv(1024 + 512, 512)  # 拼接后1536通道
        self.dec2 = DoubleConv(512 + 256, 256)   # 拼接后768通道
        self.dec3 = DoubleConv(256 + 128, 128)   # 拼接后384通道
        self.dec4 = DoubleConv(128 + 64, 64)     # 拼接后192通道
        
        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码路径
        enc1_out = self.enc1(x)
        x = self.pool(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool(enc3_out)
        
        enc4_out = self.enc4(x)
        x = self.pool(enc4_out)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码路径（跳跃连接）
        x = self.upsample(x)
        x = torch.cat([x, enc4_out], dim=1)
        x = self.dec1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.dec2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec4(x)
        
        # 输出层
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        return x


# ==================== ModelService服务层 ====================

class ModelService:
    """
    AI模型服务类 - 封装所有模型相关操作
    
    职责：
    - 模型加载和管理
    - 图像预处理（归一化、尺寸调整）
    - 模型推理（前向传播）
    - 结果后处理（转PNG）
    
    使用示例：
        >>> # 完整的分割流程
        >>> result_path = ModelService.segment_image(
        ...     image_path='uploads/image.png',
        ...     model_type='unet'
        ... )
        >>> print(f"分割结果: {result_path}")
        
        >>> # 仅加载模型
        >>> model = ModelService.load_model('unet')
        >>> if model:
        ...     print("模型加载成功")
    """
    
    # ==================== 常量定义 ====================
    
    # 默认权重路径（来自config.UNET_WEIGHT_PATH）
    DEFAULT_WEIGHT_PATH = UNET_WEIGHT_PATH
    # 不同模型的默认权重位置（未找到时回退到随机初始化）
    DEFAULT_WEIGHT_PATH_MAP = {
        "unet": UNET_WEIGHT_PATH,
        "rs_unet3_plus": os.path.join(RS_UNET3_PLUS_WEIGHT_DIR, "rs_unet3p_best.pth"),
    }
    # 默认图像尺寸（来自config.IMAGE_TARGET_SIZE）
    DEFAULT_TARGET_SIZE = IMAGE_TARGET_SIZE
    # 默认运行设备（来自config.MODEL_DEVICE）
    DEFAULT_DEVICE = MODEL_DEVICE
    # 默认模型类型（来自config.DEFAULT_MODEL_TYPE）
    DEFAULT_MODEL_ALGORITHM = DEFAULT_MODEL_TYPE
    
    # ==================== 模型加载 ====================
    
    @staticmethod
    def load_model(
        model_type: str = "unet",
        weight_path: str = None
    ) -> Optional[nn.Module]:
        """
        加载AI模型（U-Net或RS-Unet3+）
        
        加载流程：
        1. 验证模型类型
        2. 创建模型实例
        3. 加载预训练权重（如果存在）
        4. 设置为评估模式
        5. 强制使用CPU（医学影像服务器常无GPU）
        
        Args:
            model_type (str): 
              模型类型，支持：
              - 'unet': U-Net模型（通用分割）
              - 'rs_unet3_plus': RS-Unet3+模型（OCTA微血管专用，推荐）
            
            weight_path (str, optional): 
              预训练权重文件路径
              默认：./models/weights/unet_octa.pth
              如果文件不存在，使用随机初始化的模型
        
        Returns:
            Optional[nn.Module]: 
              加载好的模型实例，失败返回None
              模型已设置为评估模式（model.eval()）
              所有参数的requires_grad已设为False
        
        模型输入输出格式：
          - 输入：(batch_size, 3, 256, 256) - RGB图像
          - 输出：(batch_size, 1, 256, 256) - 分割掩码[0,1]
        
        异常处理：
          - 模型类型无效：返回None
          - 权重文件损坏：返回随机初始化模型
          - 加载过程异常：返回None
        
        示例:
            >>> # 使用默认权重
            >>> model = ModelService.load_model('unet')
            >>> if model:
            ...     print("✓ U-Net模型加载成功")
            
            >>> # 使用自定义权重
            >>> model = ModelService.load_model(
            ...     'unet',
            ...     weight_path='./custom_weights.pth'
            ... )
        """
        try:
            # ==================== 步骤1：验证模型类型 ====================
            model_type_lower = model_type.lower()
            
            if model_type_lower == 'unet':
                print(f"[INFO] 创建U-Net模型...")
                model = UNet(in_channels=3, out_channels=1)
            elif model_type_lower == 'rs_unet3_plus' or model_type_lower == 'rs-unet3+':
                print(f"[INFO] 创建RS-Unet3+模型...")
                model = RSUNet3Plus(n_channels=3, n_classes=1, base_c=64)
            else:
                print(f"[ERROR] 不支持的模型类型: {model_type}，仅支持 'unet' 或 'rs_unet3_plus'")
                return None
            
            # ==================== 步骤2：确定权重路径 ====================
            if weight_path is None:
                weight_path = ModelService.DEFAULT_WEIGHT_PATH_MAP.get(
                    model_type_lower,
                    ModelService.DEFAULT_WEIGHT_PATH
                )
            
            print(f"[INFO] 权重文件路径: {weight_path}")
            
            # ==================== 步骤3：检查权重文件 ====================
            if not os.path.exists(weight_path):
                print(f"[WARNING] 权重文件不存在: {weight_path}")
                print(f"[WARNING] 使用随机初始化的模型")
                # 不返回None，继续使用未训练的模型（用于测试）
            else:
                # ==================== 步骤4：加载权重 ====================
                try:
                    # 强制使用CPU加载
                    device = torch.device('cpu')
                    checkpoint = torch.load(weight_path, map_location=device)
                    
                    # 处理不同的权重格式
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                            print(f"[INFO] 加载权重（state_dict键）")
                        elif 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            print(f"[INFO] 加载权重（model_state_dict键）")
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                            print(f"[INFO] 加载权重（直接字典）")
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                        print(f"[INFO] 加载权重（张量格式）")
                    
                    print(f"[SUCCESS] 权重加载成功: {weight_path}")
                    
                except Exception as e:
                    print(f"[ERROR] 权重加载失败: {e}")
                    print(f"[WARNING] 继续使用随机初始化模型")
            
            # ==================== 步骤5：设置模型为评估模式 ====================
            model = model.to('cpu')  # 强制CPU
            model.eval()  # 评估模式
            
            # 禁用梯度计算
            for param in model.parameters():
                param.requires_grad = False
            
            print(f"[SUCCESS] {model_type.upper()}模型准备完成")
            print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== 图像预处理 ====================
    
    @staticmethod
    def preprocess_image(
        image_path: str,
        target_size: Tuple[int, int] = None
    ) -> Optional[torch.Tensor]:
        """
        图像预处理 - 为模型推理准备输入
        
        预处理流程：
        1. 加载图像（支持PNG/JPG/JPEG）
        2. 转换为RGB格式（去除透明通道）
        3. 调整到目标尺寸（256x256）
        4. 归一化到[0,1]范围
        5. 转换为PyTorch张量
        6. 添加batch维度
        
        Args:
            image_path (str): 
              输入图像文件路径
              支持格式：PNG, JPG, JPEG
              示例：'uploads/image.png'
            
            target_size (Tuple[int, int], optional): 
              目标尺寸 (width, height)
              默认：(256, 256)
              医学影像标准分辨率
        
        Returns:
            Optional[torch.Tensor]: 
              预处理后的张量
              - 形状：(1, 3, 256, 256)
              - 数据类型：torch.float32
              - 值范围：[0, 1]
              - 失败返回None
        
        张量维度说明：
          - dim 0: batch_size = 1（单张图像）
          - dim 1: channels = 3（RGB）
          - dim 2: height = 256
          - dim 3: width = 256
        
        异常处理：
          - 文件不存在：返回None
          - 图像损坏：返回None
          - 格式不支持：返回None
        
        示例:
            >>> # 标准用法
            >>> tensor = ModelService.preprocess_image('uploads/image.jpg')
            >>> print(tensor.shape)  # torch.Size([1, 3, 256, 256])
            
            >>> # 自定义尺寸
            >>> tensor = ModelService.preprocess_image(
            ...     'uploads/image.png',
            ...     target_size=(512, 512)
            ... )
        """
        try:
            # ==================== 步骤1：参数验证 ====================
            if not os.path.exists(image_path):
                print(f"[ERROR] 图像文件不存在: {image_path}")
                return None
            
            if target_size is None:
                target_size = ModelService.DEFAULT_TARGET_SIZE  # 来自config.IMAGE_TARGET_SIZE
            
            print(f"[INFO] 开始预处理图像: {image_path}")
            
            # ==================== 步骤2：加载并转换图像 ====================
            # PIL加载图像并转为RGB（处理RGBA透明通道）
            image = Image.open(image_path).convert('RGB')
            print(f"[INFO] 原始图像尺寸: {image.size}")
            
            # ==================== 步骤3：调整尺寸 ====================
            # BILINEAR插值：平衡质量和速度
            image = image.resize(target_size, Image.Resampling.BILINEAR)
            print(f"[INFO] 调整后尺寸: {image.size}")
            
            # ==================== 步骤4：转换为numpy数组并归一化 ====================
            # 转为numpy数组：(H, W, C)，范围[0, 255]
            image_array = np.array(image, dtype=np.float32)
            
            # 归一化到[0, 1]
            image_array = image_array / 255.0
            print(f"[INFO] 值范围: [{image_array.min():.3f}, {image_array.max():.3f}]")
            
            # ==================== 步骤5：转换为PyTorch张量 ====================
            # HWC → CHW
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            
            # 添加batch维度：CHW → BCHW
            image_tensor = image_tensor.unsqueeze(0)
            
            print(f"[SUCCESS] 预处理完成，张量形状: {image_tensor.shape}")
            
            return image_tensor
            
        except Exception as e:
            print(f"[ERROR] 图像预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== 结果后处理 ====================
    
    @staticmethod
    def postprocess_result(
        output_tensor: torch.Tensor,
        save_path: str,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Optional[str]:
        """
        结果后处理 - 将模型输出转为PNG图像
        
        后处理流程：
        1. 移除batch和channel维度
        2. 将[0,1]浮点值缩放到[0,255]整数
        3. 转换为8位灰度图（uint8）
        4. 调整到原始图像尺寸（如果提供）
        5. 保存为PNG文件
        
        Args:
            output_tensor (torch.Tensor): 
              模型输出的张量
              形状：(batch_size, 1, height, width)
              值范围：[0, 1]（Sigmoid激活后）
            
            save_path (str): 
              保存结果的完整路径
              示例：'results/image_seg.png'
              目录会自动创建
            
            original_size (Tuple[int, int], optional): 
              原始图像尺寸 (width, height)
              如果提供，会将掩码调整到该尺寸
              示例：(512, 512)
        
        Returns:
            Optional[str]: 
              成功时返回保存路径
              失败时返回None
        
        输出格式：
          - 文件格式：PNG
          - 颜色模式：灰度图（8位）
          - 值范围：0（黑色/背景）~ 255（白色/前景）
          - 压缩：无损压缩
        
        异常处理：
          - 张量维度错误：返回None
          - 保存路径无效：返回None
          - 磁盘空间不足：返回None
        
        示例:
            >>> # 标准用法
            >>> output = model(input_tensor)  # (1, 1, 256, 256)
            >>> save_path = ModelService.postprocess_result(
            ...     output,
            ...     'results/output.png'
            ... )
            >>> print(f"保存成功: {save_path}")
            
            >>> # 恢复原始尺寸
            >>> save_path = ModelService.postprocess_result(
            ...     output,
            ...     'results/output.png',
            ...     original_size=(512, 512)
            ... )
        """
        try:
            # ==================== 步骤1：移除batch和channel维度 ====================
            # (1, 1, H, W) → (H, W)
            mask = output_tensor.squeeze().detach().cpu().numpy()
            
            if mask.ndim != 2:
                print(f"[ERROR] 掩码维度异常: {mask.ndim}，预期为2")
                return None
            
            print(f"[INFO] 掩码形状: {mask.shape}")
            
            # ==================== 步骤2：缩放到[0,255] ====================
            # [0,1] → [0,255]
            mask = (mask * 255).astype(np.uint8)
            print(f"[INFO] 值范围: [{mask.min()}, {mask.max()}]")
            
            # ==================== 步骤3：调整尺寸（如果需要）====================
            if original_size is not None:
                # 使用NEAREST插值保持离散值
                mask_image = Image.fromarray(mask, mode='L')
                mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)
                mask = np.array(mask_image)
                print(f"[INFO] 掩码已调整到原始尺寸: {original_size}")
            
            # ==================== 步骤4：确保保存目录存在 ====================
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"[INFO] 创建目录: {save_dir}")
            
            # ==================== 步骤5：保存为PNG ====================
            mask_image = Image.fromarray(mask, mode='L')
            mask_image.save(save_path)
            
            print(f"[SUCCESS] 分割结果已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            print(f"[ERROR] 结果后处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== 完整的分割流程 ====================
    
    @staticmethod
    def segment_image(
        image_path: str,
        model_type: str = "unet",
        output_path: Optional[str] = None,
        weight_path: Optional[str] = None
    ) -> str:
        """
        完整的图像分割流程 - Service层核心接口
        
        完整流程：
        1. 验证输入文件
        2. 加载AI模型
        3. 预处理图像
        4. 执行模型推理
        5. 后处理结果
        6. 保存PNG文件
        7. 返回结果路径
        
        容错机制：
        - 任何步骤失败都返回原图路径
        - 不抛出异常，便于Controller处理
        - 详细的错误日志，便于调试
        
                Args:
            image_path (str): 
              输入图像路径
              支持格式：PNG, JPG, JPEG
              示例：'uploads/image.png'
            
            model_type (str): 
              模型类型
              - 'unet': U-Net模型（通用分割）
              - 'rs_unet3_plus': RS-Unet3+模型（OCTA微血管专用，推荐）
              默认：DEFAULT_MODEL_TYPE（从config加载）
            
            output_path (str, optional): 
              输出路径，如果为None则自动生成
              格式：input_filename_seg.png
              示例：'results/image_seg.png'
            
                        weight_path (str, optional):
                            覆盖默认权重路径
                            - None 时按模型类型使用默认权重搜索路径
                            - 提供路径时按指定权重加载
        
        Returns:
            str: 
              成功时返回分割结果路径
              失败时返回原图路径（容错）
        
        性能指标（256x256图像）：
          - 预处理：<100ms
          - 模型推理：1-3秒（CPU）
          - 后处理：<100ms
          - 总耗时：1-4秒
        
        示例:
            >>> # 标准用法
            >>> result = ModelService.segment_image(
            ...     'uploads/image.png',
            ...     model_type='unet'
            ... )
            >>> print(f"分割结果: {result}")
            
            >>> # Controller调用示例
            >>> class ImageController:
            ...     @staticmethod
            ...     async def segment_octa(file: UploadFile):
            ...         # ... 文件验证和保存 ...
            ...         result_path = ModelService.segment_image(
            ...             upload_path,
            ...             model_type='unet'
            ...         )
            ...         return {"result_path": result_path}
        """
        try:
            print(f"[INFO] ==================== 开始OCTA图像分割 ====================")
            print(f"[INFO] 输入图像: {image_path}")
            print(f"[INFO] 模型类型: {model_type}")
            
            # ==================== 步骤1：验证输入 ====================
            if not os.path.exists(image_path):
                print(f"[ERROR] 输入图像不存在: {image_path}")
                return image_path
            
            # ==================== 步骤2：加载模型 ====================
            print(f"[INFO] 步骤1/4: 加载AI模型...")
            model = ModelService.load_model(model_type, weight_path=weight_path)
            
            if model is None:
                print(f"[WARNING] 模型加载失败，返回原图路径")
                return image_path
            
            # ==================== 步骤3：预处理图像 ====================
            print(f"[INFO] 步骤2/4: 预处理图像...")
            input_tensor = ModelService.preprocess_image(image_path)
            
            if input_tensor is None:
                print(f"[ERROR] 图像预处理失败，返回原图路径")
                return image_path
            
            # 保存原始图像尺寸
            original_image = Image.open(image_path)
            original_size = original_image.size
            print(f"[INFO] 原始尺寸: {original_size}")
            
            # ==================== 步骤4：模型推理 ====================
            print(f"[INFO] 步骤3/4: 执行模型推理...")
            
            with torch.no_grad():  # 禁用梯度计算
                output_tensor = model(input_tensor)
            
            print(f"[INFO] 推理完成，输出形状: {output_tensor.shape}")
            
            # ==================== 步骤5：生成输出路径 ====================
            if output_path is None:
                # 自动生成：input.jpg → input_seg.png
                input_path = Path(image_path)
                output_path = str(input_path.parent / f"{input_path.stem}_seg.png")
            
            print(f"[INFO] 输出路径: {output_path}")
            
            # ==================== 步骤6：后处理并保存 ====================
            print(f"[INFO] 步骤4/4: 后处理并保存结果...")
            result_path = ModelService.postprocess_result(
                output_tensor,
                output_path,
                original_size=original_size
            )
            
            if result_path is None:
                print(f"[ERROR] 结果保存失败，返回原图路径")
                return image_path
            
            print(f"[SUCCESS] ==================== 分割完成 ====================")
            print(f"[SUCCESS] 结果路径: {result_path}")
            
            return result_path
            
        except Exception as e:
            print(f"[ERROR] 分割流程异常: {e}")
            import traceback
            traceback.print_exc()
            print(f"[WARNING] 返回原图路径以便调试")
            return image_path


# ==================== 测试代码（可选）====================

if __name__ == '__main__':
    """
    ModelService单元测试
    
    运行：python -m service.model_service
    """
    
    print("=" * 60)
    print("ModelService 单元测试")
    print("=" * 60)
    
    # 测试1：加载U-Net模型
    print("\n[测试1] 加载U-Net模型...")
    model_unet = ModelService.load_model('unet')
    if model_unet:
        print("✓ U-Net模型加载成功")
        print(f"  参数量: {sum(p.numel() for p in model_unet.parameters()):,}")
    else:
        print("✗ U-Net模型加载失败")
    
    # 测试1.5：加载RS-Unet3+模型
    print("\n[测试1.5] 加载RS-Unet3+模型...")
    model_rs = ModelService.load_model('rs_unet3_plus')
    if model_rs:
        print("✓ RS-Unet3+模型加载成功")
        print(f"  参数量: {sum(p.numel() for p in model_rs.parameters()):,}")
    else:
        print("✗ RS-Unet3+模型加载失败")
    
    # 测试2：模型前向传播
    print("\n[测试2] 测试模型推理...")
    model = model_unet if model_unet else model_rs
    if model:
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ 推理成功")
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print("\n使用说明：")
    print("1. 完整分割：ModelService.segment_image(image_path, model_type)")
    print("2. 仅加载模型：ModelService.load_model(model_type)")
    print("3. 仅预处理：ModelService.preprocess_image(image_path)")
    print("4. 仅后处理：ModelService.postprocess_result(tensor, save_path)")
    print("\n支持的模型类型：")
    print("  - 'unet': U-Net（通用分割）")
    print("  - 'rs_unet3_plus': RS-Unet3+（OCTA微血管专用）")


# ==================== 权重管理服务层 ====================

class WeightService:
    """
    模型权重管理服务 - 处理权重文件的CRUD操作
    
    职责：
    - 列出可用的权重文件
    - 获取权重文件详细信息
    - 验证权重文件合法性
    - 切换使用的模型权重
    
    架构优势：
    - 业务逻辑独立：权重管理完全解耦
    - 易于扩展：支持新增权重格式
    - 可测试性：可独立进行单元测试
    - 代码复用：多个Controller可共享
    
    使用示例：
        >>> # 列出所有权重
        >>> weights = WeightService.list_weights()
        >>> for w in weights:
        ...     print(f"{w['name']}: {w['size']}MB")
        
        >>> # 验证权重
        >>> if WeightService.validate_weight('path/to/model.pth'):
        ...     print("✓ 权重文件有效")
        
        >>> # 切换权重
        >>> WeightService.use_weight('models/weights/trained_model.pth')
    """
    
    # 当前使用的权重文件（内存缓存）
    _current_weight = None
    
    # ==================== 初始化 ====================
    
    @staticmethod
    def _get_weights_dir() -> Path:
        """
        获取权重文件目录
        
        Returns:
            Path: 权重目录路径
        """
        weights_dir = Path(__file__).parent.parent / "models" / "weights"
        
        # 确保目录存在
        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 创建权重目录: {weights_dir}")
        
        return weights_dir
    
    # ==================== 权重文件操作 ====================
    
    @staticmethod
    def get_weight_info(weight_path: Path) -> dict:
        """
        获取权重文件详细信息
        
        Args:
            weight_path: 权重文件路径（Path对象）
        
        Returns:
            dict: 权重信息字典
                - name: 文件名
                - path: 相对路径（相对于项目根目录）
                - size: 文件大小（MB）
                - modified_time: 修改时间戳
                - is_default: 是否为默认权重
                - is_valid: 是否为有效的.pth文件
        """
        try:
            stat = weight_path.stat()
            file_size_mb = round(stat.st_size / 1024 / 1024, 2)
            
            # 计算相对于项目根目录的路径
            project_root = Path(__file__).parent.parent
            relative_path = weight_path.relative_to(project_root)
            
            return {
                "name": weight_path.name,
                "path": str(relative_path),
                "size": file_size_mb,
                "modified_time": stat.st_mtime,
                "is_default": weight_path.name == "unet_octa.pth",
                "is_valid": weight_path.suffix == ".pth"
            }
        except Exception as e:
            print(f"[ERROR] 获取权重信息失败 {weight_path}: {e}")
            return None
    
    # ==================== 权重列表 ====================
    
    @staticmethod
    def list_weights() -> dict:
        """
        列出所有可用的模型权重文件
        
        Returns:
            dict: 权重列表信息
                - code: 状态码（200=成功）
                - msg: 状态信息
                - data: 数据字典
                    - weights: 权重列表
                    - count: 权重文件数量
                    - default_weight: 默认权重路径
        """
        try:
            weights_dir = WeightService._get_weights_dir()
            
            # 扫描所有 .pth 文件
            weight_files = list(weights_dir.glob("*.pth"))
            
            # 按修改时间倒序排序（最新的在前面）
            weight_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 获取权重详细信息
            weights_info = []
            for wf in weight_files:
                info = WeightService.get_weight_info(wf)
                if info:
                    weights_info.append(info)
            
            # 确定默认权重
            default_weight = None
            for w in weights_info:
                if w.get("is_default"):
                    default_weight = w["path"]
                    break
            
            if not default_weight and weights_info:
                default_weight = weights_info[0]["path"]
            
            print(f"[INFO] 扫描到 {len(weights_info)} 个权重文件")
            
            return {
                "code": 200,
                "msg": f"查询成功，共 {len(weights_info)} 个权重文件",
                "data": {
                    "weights": weights_info,
                    "count": len(weights_info),
                    "default_weight": default_weight
                }
            }
            
        except Exception as e:
            print(f"[ERROR] 列出权重文件失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "code": 500,
                "msg": f"查询权重失败: {str(e)}",
                "data": None
            }
    
    # ==================== 获取默认权重 ====================
    
    @staticmethod
    def get_default_weight() -> dict:
        """
        获取默认使用的模型权重
        
        优先级：
        1. unet_octa.pth（默认权重）
        2. 最新的训练权重
        
        Returns:
            dict: 默认权重信息
                - code: 状态码
                - msg: 状态信息
                - data: 权重详细信息（如果成功）
        """
        try:
            weights_dir = WeightService._get_weights_dir()
            default_path = weights_dir / "unet_octa.pth"
            
            # 检查默认权重是否存在
            if default_path.exists():
                info = WeightService.get_weight_info(default_path)
                if info:
                    return {
                        "code": 200,
                        "msg": "默认权重存在",
                        "data": info
                    }
            
            # 如果默认权重不存在，查找最新的权重
            weight_files = list(weights_dir.glob("*.pth"))
            if not weight_files:
                return {
                    "code": 404,
                    "msg": "未找到任何权重文件",
                    "data": None
                }
            
            weight_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_weight = weight_files[0]
            
            info = WeightService.get_weight_info(latest_weight)
            if info:
                return {
                    "code": 200,
                    "msg": "默认权重不存在，使用最新权重",
                    "data": info
                }
            
            return {
                "code": 500,
                "msg": "无法读取权重信息",
                "data": None
            }
            
        except Exception as e:
            print(f"[ERROR] 获取默认权重失败: {e}")
            return {
                "code": 500,
                "msg": f"获取默认权重失败: {str(e)}",
                "data": None
            }
    
    # ==================== 验证权重文件 ====================
    
    @staticmethod
    def validate_weight(weight_path: str) -> bool:
        """
        验证权重文件是否有效
        
        检查项：
        - 文件存在
        - 文件扩展名为 .pth
        - 文件不为空
        - 文件可读
        
        Args:
            weight_path: 权重文件路径（相对或绝对）
        
        Returns:
            bool: 权重文件是否有效
        """
        try:
            # 处理相对路径
            if weight_path.startswith("models/"):
                full_path = Path(__file__).parent.parent / weight_path
            else:
                full_path = Path(weight_path)
            
            # 检查文件存在
            if not full_path.exists():
                print(f"[WARNING] 权重文件不存在: {full_path}")
                return False
            
            # 检查文件后缀
            if full_path.suffix != ".pth":
                print(f"[WARNING] 不支持的文件格式: {full_path.suffix}")
                return False
            
            # 检查文件大小（至少1KB）
            if full_path.stat().st_size < 1024:
                print(f"[WARNING] 权重文件过小: {full_path}")
                return False
            
            # 尝试读取文件（权限检查）
            with open(full_path, 'rb') as f:
                _ = f.read(4)  # 读取前4字节检查可读性
            
            print(f"[INFO] 权重文件有效: {full_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 权重文件验证失败: {e}")
            return False
    
    # ==================== 切换权重 ====================
    
    @staticmethod
    def use_weight(weight_path: str) -> dict:
        """
        切换使用的模型权重
        
        流程：
        1. 验证权重路径有效
        2. 验证权重文件存在
        3. 验证权重文件格式
        4. 保存到内存缓存
        
        Args:
            weight_path: 权重文件路径（相对于项目根目录）
        
        Returns:
            dict: 切换结果
                - code: 状态码（200=成功）
                - msg: 状态信息
                - data: 切换后的权重信息
        """
        try:
            # ==================== 步骤1：验证权重路径 ====================
            if not weight_path or not weight_path.endswith('.pth'):
                return {
                    "code": 400,
                    "msg": "权重路径无效，必须以 .pth 结尾",
                    "data": None
                }
            
            # ==================== 步骤2：验证权重文件存在 ====================
            if not WeightService.validate_weight(weight_path):
                return {
                    "code": 400,
                    "msg": f"权重文件不存在或无效: {weight_path}",
                    "data": None
                }
            
            # ==================== 步骤3：获取权重信息 ====================
            # 构建完整路径
            if weight_path.startswith("models/"):
                full_path = Path(__file__).parent.parent / weight_path
            else:
                full_path = Path(__file__).parent.parent / "models" / "weights" / weight_path
            
            info = WeightService.get_weight_info(full_path)
            if not info:
                return {
                    "code": 500,
                    "msg": "无法读取权重信息",
                    "data": None
                }
            
            # ==================== 步骤4：保存到内存缓存 ====================
            WeightService._current_weight = weight_path
            
            print(f"[SUCCESS] 权重已切换: {weight_path}")
            
            return {
                "code": 200,
                "msg": "模型切换成功",
                "data": {
                    "current_weight": info["path"],
                    "name": info["name"],
                    "size": info["size"],
                    "is_default": info.get("is_default", False)
                }
            }
            
        except Exception as e:
            print(f"[ERROR] 切换权重失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "code": 500,
                "msg": f"切换权重失败: {str(e)}",
                "data": None
            }
    
    # ==================== 获取当前权重 ====================
    
    @staticmethod
    def get_current_weight() -> str:
        """
        获取当前使用的权重文件路径
        
        如果未设置，返回默认权重路径
        
        Returns:
            str: 权重文件路径
        """
        if WeightService._current_weight:
            return WeightService._current_weight
        
        # 默认返回 unet_octa.pth
        return "models/weights/unet_octa.pth"
