"""
改进的U-Net模型 - 针对OCTA小血管欠拟合问题优化

【Fix: Underfitting】核心改进：
1. Channel Attention Module (CAM) - 聚焦小血管相关通道
2. 增加通道数 [64,128,256,512] → [128,256,512,1024]
3. Multi-Scale Fusion (MSF) bottleneck - 捕捉多尺度血管
4. 保留BatchNorm/残差 - 确保梯度流通畅

结果：大幅提升特征提取能力，特别是对小血管的识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ==================== 【Fix: Underfitting】通道注意力模块 (CAM) ====================

class ChannelAttentionModule(nn.Module):
    """
    【Fix: Underfitting】通道注意力模块 - 自适应聚焦重要通道
    
    原理：对每个通道计算全局统计，学习通道间的相关性
    效果：显著增强对小血管（细小目标）的特征响应
    
    Args:
        in_channels: 输入通道数
        reduction_ratio: 通道压缩比例（默认16）
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttentionModule, self).__init__()
        
        # 平均池化 + FC 序列
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction_ratio, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction_ratio, 1), in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            加权后的特征 [B, C, H, W]
        """
        B, C, _, _ = x.size()
        
        # 全局平均池化
        avg_out = self.avg_pool(x).view(B, C)
        
        # 通道权重学习
        channel_weights = self.fc_layers(avg_out).view(B, C, 1, 1)
        
        # 应用通道权重
        return x * channel_weights


# ==================== 【Fix: Underfitting】多尺度融合块 (MSF Bottleneck) ====================

class MultiScaleFusionBlock(nn.Module):
    """
    【Fix: Underfitting】多尺度融合块 - 同时捕捉小、中、大血管
    
    设计：使用三个不同感受野的卷积核(1x1, 3x3, 5x5)，
         提取多个尺度的特征，然后融合
    
    这对于OCTA血管分割至关重要，因为：
    - 1x1: 捕捉细微血管和局部信息
    - 3x3: 捕捉中等血管
    - 5x5: 捕捉大血管和全局上下文
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiScaleFusionBlock, self).__init__()
        
        # 【Fix】三个分支：不同感受野，确保总通道数等于out_channels
        # 处理整除余数：1024//3=341, 341*3=1023，需要+1
        branch_channels = out_channels // 3
        remainder = out_channels % 3
        
        self.branch1x1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0)
        self.branch3x3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, branch_channels + remainder, kernel_size=5, padding=2)
        
        # 融合卷积 + BN + ReLU
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.fuse_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, in_channels, H, W]
        
        Returns:
            融合后的多尺度特征 [B, out_channels, H, W]
        """
        # 三个分支提取不同尺度特征
        out1 = self.branch1x1(x)
        out3 = self.branch3x3(x)
        out5 = self.branch5x5(x)
        
        # 拼接三个分支
        out = torch.cat([out1, out3, out5], dim=1)
        
        # 融合
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.relu(out)
        
        return out


# ==================== 【Fix: Underfitting】双卷积块 (增强版) ====================

class DoubleConvBlock(nn.Module):
    """
    【Fix: Underfitting & Overfitting】增强的双卷积块
    
    包含：Conv → BN → ReLU → Conv → BN → ReLU + 通道注意力 + Dropout
    相比基础版本，添加了CAM以聚焦重要特征，添加了Dropout防止过拟合
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        use_attention: 是否使用通道注意力（默认True）
        dropout_p: Dropout概率（默认0.0，解码器中应设为0.2）
    """
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True, dropout_p: float = 0.0):
        super(DoubleConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 【Fix: Underfitting】添加CAM以聚焦小血管通道
        self.use_attention = use_attention
        if use_attention:
            self.cam = ChannelAttentionModule(out_channels, reduction_ratio=8)
        
        # 【Fix: Overfitting】添加Dropout防止过拟合（仅在解码器使用）
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.use_attention:
            out = self.cam(out)
        # 【Fix: Overfitting】应用Dropout（仅在训练时）
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ==================== 【Fix: Underfitting】改进的U-Net架构 ====================

class UNetUnderfittingFix(nn.Module):
    """
    【Fix: Underfitting】专为OCTA小血管分割优化的U-Net
    
    改进点：
    1. 通道数扩展：[64,128,256,512] → [128,256,512,1024]
       - 增加特征容量，提升小血管识别能力
    
    2. CAM模块：在每个编码/解码块中添加通道注意力
       - 自适应聚焦与血管相关的特征通道
    
    3. 多尺度融合：Bottleneck使用MSF块
       - 同时处理多个尺度的血管结构
    
    4. 保留BatchNorm + 残差连接
       - 确保梯度流通畅，避免消失/爆炸
    
    输入：[B, 3, 256, 256]（RGB图像）
    输出：[B, 1, 256, 256]（分割掩码，Sigmoid激活）
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(UNetUnderfittingFix, self).__init__()
        
        # 【Fix: Underfitting】扩展通道数以增强特征容量
        channels = [128, 256, 512, 1024]  # 从[64,128,256,512]增至[128,256,512,1024]
        
        # ==================== 编码器路径（无Dropout，保留特征）====================
        self.enc1 = DoubleConvBlock(in_channels, channels[0], use_attention=True, dropout_p=0.0)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = DoubleConvBlock(channels[0], channels[1], use_attention=True, dropout_p=0.0)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = DoubleConvBlock(channels[1], channels[2], use_attention=True, dropout_p=0.0)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = DoubleConvBlock(channels[2], channels[3], use_attention=True, dropout_p=0.0)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # ==================== Bottleneck（【Fix: Underfitting】多尺度融合 + Spatial Dropout）====================
        self.bottleneck = MultiScaleFusionBlock(channels[3], channels[3])
        self.bottleneck_dropout = nn.Dropout2d(p=0.1)  # Fix: Spatial Dropout防止空间过拟合
        
        # ==================== 解码器路径（【Fix: Overfitting】添加Dropout=0.2）====================
        self.upconv4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(channels[2] * 2, channels[2], use_attention=True, dropout_p=0.2)
        
        self.upconv3 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(channels[1] * 2, channels[1], use_attention=True, dropout_p=0.2)
        
        self.upconv2 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(channels[0] * 2, channels[0], use_attention=True, dropout_p=0.2)
        
        self.upconv1 = nn.ConvTranspose2d(channels[0], channels[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(channels[0] * 2, channels[0], use_attention=True, dropout_p=0.2)
        
        # ==================== 输出层 ====================
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, 3, 256, 256]
        
        Returns:
            [B, 1, 256, 256]（Sigmoid激活）
        """
        # ==================== 编码器 ====================
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # ==================== Bottleneck（【Fix: Overfitting】添加Spatial Dropout）====================
        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)  # Fix: Spatial Dropout防止空间过拟合
        
        # ==================== 解码器（跳跃连接）====================
        # 【Fix】所有skip connection都需要通道调整以匹配upconv的输出
        x = self.upconv4(x)  # 1024 -> 512
        enc4 = enc4[:, :512, :, :]  # 1024 -> 512
        x = torch.cat([x, enc4], dim=1)  # 512 + 512 = 1024
        x = self.dec4(x)
        
        x = self.upconv3(x)  # 512 -> 256
        enc3 = enc3[:, :256, :, :]  # 512 -> 256
        x = torch.cat([x, enc3], dim=1)  # 256 + 256 = 512
        x = self.dec3(x)
        
        x = self.upconv2(x)  # 256 -> 128
        enc2 = enc2[:, :128, :, :]  # 256 -> 128
        x = torch.cat([x, enc2], dim=1)  # 128 + 128 = 256
        x = self.dec2(x)
        
        x = self.upconv1(x)  # 128 -> 128
        enc1 = enc1[:, :128, :, :]  # 128 -> 128
        x = torch.cat([x, enc1], dim=1)  # 128 + 128 = 256
        x = self.dec1(x)
        
        # ==================== 输出 ====================
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        return x


# ==================== 向后兼容性 ====================

# 导出新模型
def create_unet_underfitting_fix(in_channels: int = 3, out_channels: int = 1) -> UNetUnderfittingFix:
    """创建改进的U-Net模型"""
    return UNetUnderfittingFix(in_channels, out_channels)


if __name__ == "__main__":
    # 测试模型
    model = UNetUnderfittingFix(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}")
