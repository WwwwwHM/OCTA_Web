"""
OCTA图像分割预测服务

Fix: 平台优化 - 核心预测功能（集成所有core模块）

功能：
1. 加载模型权重（core.model_loader）
2. 图像预处理（core.data_process）
3. 模型推理（设备自适应）
4. 结果后处理（core.data_process）
5. Base64编码/本地保存

特点：
- 完全对齐本地baseline脚本（预处理参数100%一致）
- 自动设备适配（GPU优先，无GPU则CPU）
- 详细的日志记录（推理耗时、设备信息）
- 容错机制（任何错误都有明确提示）

作者：OCTA Web项目组
日期：2026-01-27
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from fastapi import HTTPException, status

from config.config import DEFAULT_DEVICE, MODEL_DEVICE
from core.model_loader import get_loader
from core.data_process import get_processor
from models.unet import UNetUnderfittingFix
# from service.weight_service import WeightService  # 已废弃，使用router/weight_router.py

logger = logging.getLogger(__name__)


class PredictionService:
    """OCTA图像分割预测服务"""
    
    def __init__(self):
        """初始化预测服务"""
        # 设备自适应
        self.device = self._select_device()
        logger.info(f"[预测服务] 初始化完成，设备: {self.device}")
        
        # 获取全局组件
        self.model_loader = get_loader(device=self.device)
        self.processor = get_processor()
        
        # 模型缓存（避免重复加载）
        self._cached_model = None
        self._cached_weight_id = None
    
    def _select_device(self) -> str:
        """
        自动选择推理设备
        
        Returns:
            设备类型（'cuda'/'cpu'）
        """
        # 优先使用配置文件指定的设备
        if MODEL_DEVICE and MODEL_DEVICE != 'auto':
            if MODEL_DEVICE == 'cuda' and torch.cuda.is_available():
                logger.info("[设备选择] ✓ 使用配置指定的CUDA")
                return 'cuda'
            elif MODEL_DEVICE == 'cpu':
                logger.info("[设备选择] ✓ 使用配置指定的CPU")
                return 'cpu'
        
        # 自动选择：GPU优先
        if torch.cuda.is_available():
            logger.info("[设备选择] ✓ GPU可用，优先使用CUDA")
            return 'cuda'
        else:
            logger.info("[设备选择] ⚠ GPU不可用，使用CPU")
            return 'cpu'
    
    def _load_model_with_cache(
        self,
        weight_id: Optional[str],
        model_type: str = 'unet'
    ) -> torch.nn.Module:
        """
        加载模型（带缓存）
        
        Args:
            weight_id: 权重ID
            model_type: 模型类型
        
        Returns:
            加载好的模型
        """
        # 检查缓存
        if self._cached_model is not None and self._cached_weight_id == weight_id:
            logger.debug(f"[模型加载] ✓ 使用缓存模型，weight_id={weight_id}")
            return self._cached_model
        
        # 创建模型实例
        if model_type == 'unet':
            model = UNetUnderfittingFix(in_channels=3, out_channels=1)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的模型类型: {model_type}"
            )
        
        # 解析权重路径
        weight_path = WeightService.resolve_weight_path(weight_id, model_type)
        weight_path_obj = Path(weight_path)
        
        # 加载权重
        success, error_msg, loaded_model = self.model_loader.load_model(
            model=model,
            weight_path=weight_path_obj,
            strict=False
        )
        
        if not success:
            logger.error(f"[模型加载] ✗ 失败: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"模型加载失败: {error_msg}"
            )
        
        # 更新缓存
        self._cached_model = loaded_model
        self._cached_weight_id = weight_id
        
        return loaded_model
    
    def predict(
        self,
        image_path: Path,
        weight_id: Optional[str] = None,
        model_type: str = 'unet',
        save_result: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        执行OCTA图像分割预测
        
        Args:
            image_path: 输入图像路径
            weight_id: 权重ID（可选，默认官方权重）
            model_type: 模型类型（默认unet）
            save_result: 是否保存结果到本地
            output_dir: 输出目录（save_result=True时必需）
        
        Returns:
            预测结果字典：
            {
                'mask_base64': str,  # Base64编码的掩码
                'mask_path': str,    # 本地保存路径（如果save_result=True）
                'inference_time': float,  # 推理耗时（秒）
                'device': str,       # 运行设备
                'model_type': str,   # 模型类型
                'weight_id': str,    # 权重ID
                'image_size': tuple, # 原始图像尺寸
            }
        """
        logger.info(f"[预测] 开始处理图像: {image_path}")
        start_time = time.time()
        
        try:
            # 步骤1：加载模型
            model = self._load_model_with_cache(weight_id, model_type)
            
            # 步骤2：图像预处理
            input_tensor, original_size = self.processor.preprocess(
                image_path=image_path,
                device=self.device
            )
            logger.info(f"[预测] ✓ 预处理完成，张量形状={input_tensor.shape}, 原始尺寸={original_size}")
            
            # 步骤3：模型推理
            inference_start = time.time()
            with torch.no_grad():
                output_tensor = model(input_tensor)
            inference_time = time.time() - inference_start
            logger.info(f"[预测] ✓ 推理完成，耗时={inference_time:.3f}秒")
            
            # 步骤4：结果后处理
            mask_array = self.processor.postprocess(
                output_tensor=output_tensor,
                original_size=original_size
            )
            logger.info(f"[预测] ✓ 后处理完成，掩码形状={mask_array.shape}")
            
            # 步骤5：Base64编码
            mask_base64 = self.processor.mask_to_base64(mask_array)
            
            # 步骤6：本地保存（可选）
            mask_path = None
            if save_result and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / f"{image_path.stem}_seg.png"
                self.processor.save_mask(mask_array, mask_path)
            
            total_time = time.time() - start_time
            logger.info(f"[预测] ✓ 完成，总耗时={total_time:.3f}秒")
            
            # 获取设备信息
            device_info = self.model_loader.get_device_info()
            
            return {
                'mask_base64': mask_base64,
                'mask_path': str(mask_path) if mask_path else None,
                'inference_time': round(inference_time, 3),
                'total_time': round(total_time, 3),
                'device': device_info.get('device', self.device),
                'cuda_info': device_info if device_info.get('cuda_available') else None,
                'model_type': model_type,
                'weight_id': weight_id or 'official',
                'image_size': original_size,
            }
            
        except Exception as e:
            logger.error(f"[预测] ✗ 失败: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"预测失败: {str(e)}"
            )


# 全局预测服务实例
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """获取全局预测服务实例"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
