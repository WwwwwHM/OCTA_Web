"""
血管分割推理脚本（带形态学后处理）

功能：
1. 加载训练好的模型进行推理
2. 形态学后处理（细化血管边缘，去除噪声）
3. 边缘锐化（提升血管边界清晰度）
4. 保存高质量结果
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.unet_underfitting_fix import UNetUnderfittingFix


# ==================== 配置 ====================

class InferenceConfig:
    """推理配置"""
    MODEL_PATH = "models/weights_vessel_enhanced/best_model.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SIZE = 256
    THRESHOLD = 0.5  # 二值化阈值
    
    # 形态学操作参数
    MORPH_KERNEL_SIZE = 3  # 形态学核大小
    OPEN_ITERATIONS = 1    # 开运算迭代次数（去除小噪声）
    CLOSE_ITERATIONS = 1   # 闭运算迭代次数（填补小空洞）
    
    # 边缘锐化参数
    SHARPEN_STRENGTH = 1.5  # 锐化强度


# ==================== 推理器类 ====================

class VesselInference:
    """血管分割推理器"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = UNetUnderfittingFix(in_channels=3, out_channels=1).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ 最佳Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"  ✓ 模型已加载到 {device}")
        
        # 数据预处理
        self.transform = A.Compose([
            A.Resize(InferenceConfig.IMAGE_SIZE, InferenceConfig.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            (处理后的张量, 原始尺寸)
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 转换为numpy数组
        image_np = np.array(image)
        
        # 应用变换
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)  # [1, 3, H, W]
        
        return image_tensor, original_size
    
    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        模型推理
        
        Args:
            image_tensor: 输入张量 [1, 3, H, W]
        
        Returns:
            预测掩码 [H, W]，范围[0, 1]
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)  # [1, 1, H, W]
            
            # 转换为numpy数组
            mask = output.squeeze().cpu().numpy()  # [H, W]
        
        return mask
    
    def morphological_postprocess(self, mask: np.ndarray, 
                                  kernel_size: int = 3,
                                  open_iter: int = 1,
                                  close_iter: int = 1) -> np.ndarray:
        """
        形态学后处理
        
        目标：
        1. 去除小噪声点（开运算）
        2. 填补血管内部小空洞（闭运算）
        3. 保持血管边缘清晰
        
        Args:
            mask: 输入掩码 [H, W]，范围[0, 1]
            kernel_size: 结构元素大小
            open_iter: 开运算迭代次数
            close_iter: 闭运算迭代次数
        
        Returns:
            处理后的掩码
        """
        # 二值化
        binary_mask = (mask > InferenceConfig.THRESHOLD).astype(np.uint8) * 255
        
        # 创建结构元素（椭圆形，更适合血管）
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        # 开运算：先腐蚀后膨胀，去除小噪声点
        if open_iter > 0:
            binary_mask = cv2.morphologyEx(
                binary_mask, 
                cv2.MORPH_OPEN, 
                kernel, 
                iterations=open_iter
            )
        
        # 闭运算：先膨胀后腐蚀，填补小空洞
        if close_iter > 0:
            binary_mask = cv2.morphologyEx(
                binary_mask, 
                cv2.MORPH_CLOSE, 
                kernel, 
                iterations=close_iter
            )
        
        return binary_mask
    
    def edge_sharpening(self, mask: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        边缘锐化
        
        使用Unsharp Masking技术增强血管边缘
        
        Args:
            mask: 输入掩码 [H, W]
            strength: 锐化强度（1.0-3.0）
        
        Returns:
            锐化后的掩码
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(mask, (0, 0), 3.0)
        
        # Unsharp Masking: 原图 + strength * (原图 - 模糊图)
        sharpened = cv2.addWeighted(
            mask, 1.0 + strength,
            blurred, -strength,
            0
        )
        
        # 裁剪到[0, 255]
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def remove_small_components(self, mask: np.ndarray, 
                               min_size: int = 50) -> np.ndarray:
        """
        去除小连通域（噪声点）
        
        Args:
            mask: 输入掩码 [H, W]，二值图（0或255）
            min_size: 最小保留的连通域面积
        
        Returns:
            处理后的掩码
        """
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # 创建输出掩码
        output = np.zeros_like(mask)
        
        # 保留大于min_size的连通域（跳过背景标签0）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output[labels == i] = 255
        
        return output
    
    def postprocess(self, mask: np.ndarray, 
                   original_size: Tuple[int, int],
                   apply_morphology: bool = True,
                   apply_sharpening: bool = True,
                   remove_noise: bool = True) -> np.ndarray:
        """
        完整后处理流程
        
        Args:
            mask: 模型输出掩码 [H, W]，范围[0, 1]
            original_size: 原始图像尺寸 (width, height)
            apply_morphology: 是否应用形态学操作
            apply_sharpening: 是否应用边缘锐化
            remove_noise: 是否去除小噪声
        
        Returns:
            处理后的掩码 [H, W]，范围[0, 255]
        """
        # 1. 形态学处理
        if apply_morphology:
            mask = self.morphological_postprocess(
                mask,
                kernel_size=InferenceConfig.MORPH_KERNEL_SIZE,
                open_iter=InferenceConfig.OPEN_ITERATIONS,
                close_iter=InferenceConfig.CLOSE_ITERATIONS
            )
        else:
            # 直接二值化
            mask = (mask > InferenceConfig.THRESHOLD).astype(np.uint8) * 255
        
        # 2. 去除小噪声
        if remove_noise:
            mask = self.remove_small_components(mask, min_size=50)
        
        # 3. 调整到原始尺寸
        mask = cv2.resize(
            mask, 
            original_size, 
            interpolation=cv2.INTER_NEAREST  # 保持二值特性
        )
        
        # 4. 边缘锐化（可选）
        if apply_sharpening:
            mask = self.edge_sharpening(
                mask, 
                strength=InferenceConfig.SHARPEN_STRENGTH
            )
        
        return mask
    
    def infer(self, image_path: str, 
             output_path: Optional[str] = None,
             apply_postprocess: bool = True) -> np.ndarray:
        """
        完整推理流程
        
        Args:
            image_path: 输入图像路径
            output_path: 输出路径（None则自动生成）
            apply_postprocess: 是否应用后处理
        
        Returns:
            分割结果掩码
        """
        print(f"推理: {image_path}")
        
        # 预处理
        image_tensor, original_size = self.preprocess(image_path)
        
        # 推理
        mask = self.predict(image_tensor)
        
        # 后处理
        if apply_postprocess:
            mask = self.postprocess(mask, original_size)
        else:
            # 简单二值化 + 调整尺寸
            mask = (mask > InferenceConfig.THRESHOLD).astype(np.uint8) * 255
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # 保存结果
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_seg.png")
        
        cv2.imwrite(output_path, mask)
        print(f"  ✓ 结果已保存: {output_path}")
        
        return mask


# ==================== 批量推理 ====================

def batch_inference(input_dir: str, output_dir: str, model_path: str):
    """
    批量推理
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出掩码目录
        model_path: 模型权重路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化推理器
    device = InferenceConfig.DEVICE
    inferencer = VesselInference(model_path, device)
    
    # 获取所有图像
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    
    print(f"\n批量推理: {len(image_files)} 张图像")
    print("=" * 80)
    
    # 逐个推理
    for idx, image_file in enumerate(image_files, 1):
        output_file = Path(output_dir) / f"{image_file.stem}_seg.png"
        
        print(f"[{idx}/{len(image_files)}] {image_file.name}")
        inferencer.infer(str(image_file), str(output_file))
    
    print("=" * 80)
    print("批量推理完成！")


# ==================== 主函数 ====================

def main():
    """主函数（单图推理示例）"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  单图推理: python inference_vessel_enhanced.py <image_path>")
        print("  批量推理: python inference_vessel_enhanced.py <input_dir> <output_dir>")
        return
    
    if len(sys.argv) == 2:
        # 单图推理
        image_path = sys.argv[1]
        inferencer = VesselInference(
            InferenceConfig.MODEL_PATH, 
            InferenceConfig.DEVICE
        )
        inferencer.infer(image_path)
    
    elif len(sys.argv) == 3:
        # 批量推理
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        batch_inference(input_dir, output_dir, InferenceConfig.MODEL_PATH)


if __name__ == "__main__":
    main()
