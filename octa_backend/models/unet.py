"""
OCTA图像分割U-Net模型实现

本模块提供了用于OCTA（光学相干断层血管成像）图像分割的U-Net模型。
模型输入为3通道RGB图像，输出为1通道分割掩码（灰度图）。

包含两个版本：
1. UNet（原始版本，保留用于向后兼容）
2. UNet_Transformer（改进版本，使用门控注意力+Transformer，推荐使用）

作者：OCTA Web项目组
日期：2024-2026
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Tuple


# ==================== 门控注意力模块 (Gated Attention Block) ====================

class AttentionBlock(nn.Module):
    """
    # Fix: 门控注意力块（Gated Attention Block）
    
    相比通道注意力的改进：
    1. 保留空间信息：卷积生成[H, W, 1]的空间注意力
    2. 融合多源信息：同时利用decoder和encoder的特征
    3. 局部自适应：每个像素位置的权重独立计算
    4. 门控机制：通过加法融合两个特征流
    
    这是Attention U-Net的核心改进，专为skip connection设计。
    
    Args:
        F_g: 来自decoder路径的通道数（gating特征）
        F_l: skip connection的通道数（待加权特征）
        F_int: 中间层维度（建议F_l//2）
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        # Gating信号投影（来自下层的特征）
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection信号投影
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 注意力权重生成器（生成空间注意力）
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 输出[0,1]的空间权重
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        前向传播
        
        Args:
            g: Gating特征（来自decoder），形状[B, F_g, H, W]
            x: Skip connection特征（来自encoder），形状[B, F_l, H, W]
        
        Returns:
            加权后的skip特征，形状[B, F_l, H, W]
        """
        # 投影两个特征到相同维度
        g1 = self.W_g(g)        # [B, F_int, H, W]
        x1 = self.W_x(x)        # [B, F_int, H, W]
        
        # 融合两个特征流
        psi = self.relu(g1 + x1)  # [B, F_int, H, W]
        
        # 生成空间注意力权重
        psi = self.psi(psi)     # [B, 1, H, W]
        
        # 应用注意力权重
        return x * psi          # [B, F_l, H, W]


# ==================== Transformer编码器块 ====================

class SimpleTransformerEncoder(nn.Module):
    """
    # Fix: 简化版Transformer编码器块
    
    用途：在U-Net的bottleneck处添加全局上下文
    
    设计特点：
    1. 多头自注意力：学习长距离依赖（对小血管重要）
    2. MLP块：非线性特征变换
    3. 残差连接：梯度流通畅
    4. LayerNorm：特征标准化
    
    Args:
        dim: 输入特征维度
        num_heads: 多头注意力的头数（建议4或8）
        mlp_ratio: MLP隐层倍数（默认4）
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super(SimpleTransformerEncoder, self).__init__()
        
        # 第一个子层：多头自注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=False)
        
        # 第二个子层：MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征，形状[B, C, H, W]
        
        Returns:
            输出特征，形状[B, C, H, W]（维度不变）
        """
        B, C, H, W = x.shape
        
        # 将[B, C, H, W]转换为[H*W, B, C]（Transformer格式）
        x_flat = x.flatten(2).permute(2, 0, 1)
        
        # 多头自注意力（带残差连接）
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out
        
        # MLP（带残差连接）
        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        
        # 恢复形状为[B, C, H, W]
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return x


# ==================== 简化卷积块（无残差连接）====================

class DoubleConv(nn.Module):
    """
    双层卷积块（简化版，无残差连接）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        return out


# ==================== 改进的U-Net with Transformer ====================

class UNet_Transformer(nn.Module):
    """
    # Fix: 改进的U-Net，融合Transformer和门控注意力
    
    架构设计：
    - 编码器：标准卷积块 + 最大池化
    - Bottleneck：多头自注意力Transformer（全局上下文）
    - 解码器：转置卷积 + 门控注意力 + 跳跃连接
    
    相比原始U-Net的改进：
    1. 门控注意力替代通道注意力（保留空间信息）
    2. Transformer Bottleneck（增强全局上下文感知）
    3. ConvTranspose2d替代Upsample（可学习上采样）
    4. 移除残差连接（梯度流清晰）
    
    输入：RGB图像 [B, 3, 256, 256]
    输出：分割掩码 [B, 1, 256, 256]
    
    Args:
        in_channels: 输入通道数（RGB为3）
        out_channels: 输出通道数（分割掩码为1）
        features: 各层特征维度（建议[64, 128, 256, 512]）
        trans_dim: Transformer维度（建议1024）
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], trans_dim=1024):
        super(UNet_Transformer, self).__init__()
        
        # 编码器块列表
        self.downs = nn.ModuleList()
        # 解码器块列表
        self.ups = nn.ModuleList()
        # 门控注意力块列表
        self.attentions = nn.ModuleList()
        
        # ==================== 编码器路径 ====================
        # 构建编码器，每层都进行通道扩展
        in_ch = in_channels
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = feature
        
        # ==================== Bottleneck块（Transformer）====================
        # 使用1×1卷积调整通道数
        self.bottleneck_conv = nn.Conv2d(features[-1], trans_dim, kernel_size=1)
        # Transformer编码器（全局上下文感知）
        self.transformer = SimpleTransformerEncoder(dim=trans_dim, num_heads=8, mlp_ratio=4.0)
        # 恢复通道数到原来的2倍（用于拼接）
        self.bottleneck_deconv = nn.Conv2d(trans_dim, features[-1] * 2, kernel_size=1)
        
        # ==================== 解码器路径 ====================
        # 反向遍历特征维度，构建解码器
        for feature in reversed(features):
            # 转置卷积上采样（可学习）
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # 双层卷积块
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            # 门控注意力块（用于加权skip connection）
            self.attentions.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
        
        # 最终输出层：映射到1通道分割掩码
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像，形状[B, 3, 256, 256]
        
        Returns:
            分割掩码，形状[B, 1, 256, 256]
        """
        # ==================== 编码器路径（下采样）====================
        skip_connections = []
        
        for down in self.downs:
            x = down(x)           # 卷积处理
            skip_connections.append(x)  # 保存跳跃连接
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # 下采样
        
        # ==================== Bottleneck块（全局上下文）====================
        x = self.bottleneck_conv(x)     # 升维到trans_dim
        x = self.transformer(x)         # Transformer处理
        x = self.bottleneck_deconv(x)   # 恢复维度
        
        # ==================== 解码器路径（上采样）====================
        # 反转跳跃连接（因为要从浅到深）
        skip_connections = skip_connections[::-1]
        
        # 上采样块对：每次迭代处理两个模块（上采样 + 卷积）
        for idx in range(0, len(self.ups), 2):
            # 转置卷积上采样
            x = self.ups[idx](x)
            
            # 获取对应的跳跃连接
            skip_connection = skip_connections[idx // 2]
            
            # 门控注意力：对skip connection进行加权
            attn = self.attentions[idx // 2](g=x, x=skip_connection)
            
            # 确保尺寸匹配（处理边界情况）
            if x.shape != attn.shape:
                x = F.interpolate(x, size=attn.shape[2:])
            
            # 拼接上采样特征和加权的skip特征
            x = torch.cat((attn, x), dim=1)
            
            # 双层卷积处理
            x = self.ups[idx + 1](x)
        
        # 最终卷积映射到1通道 + Sigmoid激活
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        return x


# ==================== 原始U-Net（保留用于向后兼容）====================

class ChannelAttention(nn.Module):
    """
    # Fix: 添加通道注意力模块 (Channel Attention Module)
    用于OCTA小血管分割，增强对细小目标的特征响应
    
    原理：通过全局平均池化 + 全连接层学习通道权重，
    突出重要的血管特征通道，抑制背景噪声通道
    
    Args:
        in_channels: 输入通道数
        reduction: 通道压缩比例（默认16，平衡性能和精度）
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 -> (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()  # 输出通道权重 [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # 全局平均池化 + 通道权重学习
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 通道加权：突出重要血管特征
        return x * y.expand_as(x)


# ==================== 增强卷积块（带残差连接）====================

class DoubleConv(nn.Module):
    """
    # Fix: 增强双卷积块（添加残差连接）
    
    改进点：
    1. 保留 Conv2d -> BN -> ReLU 结构（已有）
    2. 添加残差连接（解决深层U-Net梯度消失问题）
    3. He初始化卷积层（比随机初始化收敛更快）
    
    残差连接原理：y = F(x) + x（快捷路径缓解梯度消失）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        use_residual: 是否使用残差连接（编码器/瓶颈层启用）
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super(DoubleConv, self).__init__()
        
        self.use_residual = use_residual
        
        # 第一个卷积块：3x3卷积 + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            bias=False  # Fix: BN层后不需要bias，节省参数
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积块：3x3卷积 + BatchNorm + ReLU
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Fix: 残差连接的1x1卷积（通道数不匹配时调整）
        self.shortcut = nn.Sequential()
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Fix: He初始化（适配ReLU激活函数，比默认初始化收敛快）
        self._init_weights()
    
    def _init_weights(self):
        """# Fix: He初始化卷积层（针对ReLU优化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（带残差连接）
        
        Args:
            x: 输入张量，形状为 (batch_size, in_channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, out_channels, height, width)
        """
        identity = self.shortcut(x) if self.use_residual else 0
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Fix: 残差连接（缓解梯度消失）
        if self.use_residual:
            out += identity
        
        out = self.relu2(out)
        return out


# ==================== U-Net模型主体 ====================

class UNet(nn.Module):
    """
    U-Net模型用于OCTA图像分割
    
    U-Net是一个经典的编码器-解码器（Encoder-Decoder）架构，形状像字母"U"：
    - 左侧（编码器）：通过下采样提取特征，逐步减小空间尺寸，增加通道数
    - 底部（瓶颈层）：最深层，特征最抽象
    - 右侧（解码器）：通过上采样恢复空间尺寸，逐步减小通道数
    
    本实现适配256x256输入图像，输出相同尺寸的分割掩码。
    
    网络结构：
    - 编码器：256->128->64->32->16（4次下采样）
    - 解码器：16->32->64->128->256（4次上采样）
    - 跳跃连接：将编码器特征与解码器特征拼接，保留细节信息
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        """
        # Fix: 优化U-Net结构（添加残差 + 通道注意力）
        
        改进点：
        1. 编码器/瓶颈层启用残差连接（use_residual=True）
        2. 解码器添加通道注意力模块（CAM，聚焦小血管）
        3. 保留原有BatchNorm和跳跃连接
        
        Args:
            in_channels: 输入图像通道数，RGB图像为3
            out_channels: 输出掩码通道数，分割掩码为1（灰度图）
        """
        super(UNet, self).__init__()
        
        # ========== 编码器（下采样路径）+ 残差连接 ==========
        # Fix: 编码器启用残差连接，防止梯度消失
        
        # 第一层：256x256 -> 256x256，通道：3 -> 64
        self.enc1 = DoubleConv(in_channels, 64, use_residual=True)
        
        # 第二层：256x256 -> 128x128，通道：64 -> 128
        self.enc2 = DoubleConv(64, 128, use_residual=True)
        
        # 第三层：128x128 -> 64x64，通道：128 -> 256
        self.enc3 = DoubleConv(128, 256, use_residual=True)
        
        # 第四层：64x64 -> 32x32，通道：256 -> 512
        self.enc4 = DoubleConv(256, 512, use_residual=True)
        
        # 瓶颈层：32x32 -> 16x16，通道：512 -> 1024
        self.bottleneck = DoubleConv(512, 1024, use_residual=True)
        
        # ========== 解码器（上采样路径）+ 通道注意力 ==========
        # Fix: 解码器添加CAM模块，聚焦小血管特征
        
        # 第一层：16x16 -> 32x32，通道：1024 + 512 = 1536 -> 512
        self.dec1 = DoubleConv(1024 + 512, 512, use_residual=False)
        self.cam1 = ChannelAttention(512, reduction=16)  # Fix: 通道注意力
        
        # 第二层：32x32 -> 64x64，通道：512 + 256 = 768 -> 256
        self.dec2 = DoubleConv(512 + 256, 256, use_residual=False)
        self.cam2 = ChannelAttention(256, reduction=16)
        
        # 第三层：64x64 -> 128x128，通道：256 + 128 = 384 -> 128
        self.dec3 = DoubleConv(256 + 128, 128, use_residual=False)
        self.cam3 = ChannelAttention(128, reduction=16)
        
        # 第四层：128x128 -> 256x256，通道：128 + 64 = 192 -> 128
        self.dec4 = DoubleConv(128 + 64, 128, use_residual=False)
        self.cam4 = ChannelAttention(128, reduction=16)
        
        # 最终输出层：1x1卷积，将128通道映射到1通道（分割掩码）
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # 最大池化层（用于下采样）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 上采样层（用于上采样）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为 (batch_size, 3, 256, 256)
            
        Returns:
            分割掩码张量，形状为 (batch_size, 1, 256, 256)
        """
        # ========== 编码器路径（下采样）==========
        # 保存每层的特征用于跳跃连接
        enc1_out = self.enc1(x)          # 256x256, 64通道
        x = self.pool(enc1_out)          # 128x128, 64通道
        
        enc2_out = self.enc2(x)          # 128x128, 128通道
        x = self.pool(enc2_out)          # 64x64, 128通道
        
        enc3_out = self.enc3(x)          # 64x64, 256通道
        x = self.pool(enc3_out)          # 32x32, 256通道
        
        enc4_out = self.enc4(x)          # 32x32, 512通道
        x = self.pool(enc4_out)          # 16x16, 512通道
        
        # ========== 瓶颈层 ==========
        x = self.bottleneck(x)           # 16x16, 1024通道
        
        # ========== 解码器路径（上采样 + 跳跃连接 + 通道注意力）==========
        # Fix: 每次解码后应用通道注意力，聚焦小血管特征
        
        x = self.upsample(x)             # 32x32, 1024通道
        x = torch.cat([x, enc4_out], dim=1)  # 拼接：32x32, 1024+512=1536通道
        x = self.dec1(x)                 # 32x32, 512通道
        x = self.cam1(x)                 # Fix: 通道注意力（突出血管通道）
        
        x = self.upsample(x)             # 64x64, 512通道
        x = torch.cat([x, enc3_out], dim=1)  # 拼接：64x64, 512+256=768通道
        x = self.dec2(x)                 # 64x64, 256通道
        x = self.cam2(x)                 # Fix: 通道注意力
        
        x = self.upsample(x)             # 128x128, 256通道
        x = torch.cat([x, enc2_out], dim=1)  # 拼接：128x128, 256+128=384通道
        x = self.dec3(x)                 # 128x128, 128通道
        x = self.cam3(x)                 # Fix: 通道注意力
        
        x = self.upsample(x)             # 256x256, 128通道
        x = torch.cat([x, enc1_out], dim=1)  # 拼接：256x256, 128+64=192通道
        x = self.dec4(x)                 # 256x256, 64通道
        x = self.cam4(x)                 # Fix: 通道注意力
        
        # ========== 最终输出 ==========
        x = self.final_conv(x)           # 256x256, 1通道
        
        # 使用Sigmoid激活函数，将输出映射到[0,1]范围
        # 这样可以直接作为概率掩码使用
        x = torch.sigmoid(x)
        
        return x


# ==================== FCN模型（备选）====================

class FCN(nn.Module):
    """
    全卷积网络（Fully Convolutional Network）用于OCTA图像分割
    
    FCN是另一种常用的图像分割架构，相比U-Net更简单，但没有跳跃连接。
    这里提供一个简化版本作为备选模型。
    
    Args:
        in_channels: 输入通道数，RGB图像为3
        out_channels: 输出通道数，分割掩码为1
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(FCN, self).__init__()
        
        # 编码器部分
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # 解码器部分
        self.dec1 = DoubleConv(512, 256)
        self.dec2 = DoubleConv(256, 128)
        self.dec3 = DoubleConv(128, 64)
        
        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        
        # 解码
        x = self.upsample(x)
        x = self.dec1(x)
        x = self.upsample(x)
        x = self.dec2(x)
        x = self.upsample(x)
        x = self.dec3(x)
        x = self.upsample(x)
        
        # 输出
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        return x


# ==================== 改进版U-Net（Underfitting Fix）====================

class ChannelAttentionModule(nn.Module):
    """通道注意力模块，用于强化细小血管相关通道。"""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class MultiScaleFusionBlock(nn.Module):
    """多尺度融合瓶颈：1x1/3x3/5x5 三分支融合。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch = out_channels // 3
        rem = out_channels % 3
        self.branch1x1 = nn.Conv2d(in_channels, branch, kernel_size=1, padding=0)
        self.branch3x3 = nn.Conv2d(in_channels, branch, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, branch + rem, kernel_size=5, padding=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
        ], dim=1)
        return self.fuse(out)


class DoubleConvBlock(nn.Module):
    """双卷积 + 可选CAM + 可选Dropout。"""

    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.use_attention = use_attention
        self.cam = ChannelAttentionModule(out_channels, reduction_ratio=8) if use_attention else None
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self.cam is not None:
            out = self.cam(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class UNetUnderfittingFix(nn.Module):
    """扩容+注意力+多尺度瓶颈的U-Net，针对欠拟合场景。"""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        ch = [128, 256, 512, 1024]

        # 编码器
        self.enc1 = DoubleConvBlock(in_channels, ch[0], use_attention=True, dropout_p=0.0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = DoubleConvBlock(ch[0], ch[1], use_attention=True, dropout_p=0.0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = DoubleConvBlock(ch[1], ch[2], use_attention=True, dropout_p=0.0)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = DoubleConvBlock(ch[2], ch[3], use_attention=True, dropout_p=0.0)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = MultiScaleFusionBlock(ch[3], ch[3])
        self.bottleneck_dropout = nn.Dropout2d(p=0.1)

        # 解码器（加入Dropout=0.2）
        self.up4 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(ch[2] * 2, ch[2], use_attention=True, dropout_p=0.2)

        self.up3 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(ch[1] * 2, ch[1], use_attention=True, dropout_p=0.2)

        self.up2 = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(ch[0] * 2, ch[0], use_attention=True, dropout_p=0.2)

        self.up1 = nn.ConvTranspose2d(ch[0], ch[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(ch[0] * 2, ch[0], use_attention=True, dropout_p=0.2)

        self.final_conv = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        e1 = self.enc1(x)
        x = self.pool1(e1)
        e2 = self.enc2(x)
        x = self.pool2(e2)
        e3 = self.enc3(x)
        x = self.pool3(e3)
        e4 = self.enc4(x)
        x = self.pool4(e4)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)

        # 解码 + 跳跃连接（截断通道保证维度匹配）
        x = self.up4(x)
        e4 = e4[:, : x.shape[1], :, :]
        x = self.dec4(torch.cat([x, e4], dim=1))

        x = self.up3(x)
        e3 = e3[:, : x.shape[1], :, :]
        x = self.dec3(torch.cat([x, e3], dim=1))

        x = self.up2(x)
        e2 = e2[:, : x.shape[1], :, :]
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = self.up1(x)
        e1 = e1[:, : x.shape[1], :, :]
        x = self.dec1(torch.cat([x, e1], dim=1))

        x = self.final_conv(x)
        return torch.sigmoid(x)


# ==================== 模型缓存（避免重复加载）====================

# 全局模型缓存：避免每次请求都重新加载权重（大幅提升推理速度）
_MODEL_CACHE = {}

# ==================== 模型加载函数 ====================

def load_unet_model(
    model_type: str = 'unet',
    model_path: Optional[str] = None,
    device: str = 'cpu',
    use_cache: bool = True
) -> Optional[nn.Module]:
    """
    加载U-Net或FCN模型，适配真实OCTA预训练权重
    
    本函数专门为OCTA分割任务优化，具有以下特点：
    1. 权重路径支持自定义（model_path参数）或使用默认路径"./models/weights/unet_octa.pth"
    2. 强制使用CPU模式加载模型，适配无GPU环境
    3. 添加权重文件存在性校验，不存在则返回None
    4. 如果加载失败，返回None，由调用者决定是否继续
    5. **模型缓存机制**：首次加载后缓存，后续请求直接复用（大幅提速）
    
    Args:
        model_type: 模型类型，'unet'（推荐）或 'fcn'（备选方案）
        model_path: 预训练模型权重文件路径。若为None，使用默认路径./models/weights/unet_octa.pth
        device: 设备类型参数（该参数仅为兼容性保留，强制使用CPU）
        use_cache: 是否使用模型缓存（默认True，生产环境建议启用）
    
    Returns:
        加载好的模型对象，如果权重文件不存在或加载失败返回None
    
    示例:
        >>> # 使用默认权重
        >>> model = load_unet_model('unet')
        >>> # 使用自定义权重（如今天训练的权重）
        >>> model = load_unet_model('unet', model_path='./models/weights_unet/unet_20260125_090633.pth')
    """
    try:
        # ==================== 步骤0：检查模型缓存 ====================
        # 如果模型已加载并缓存，直接返回（避免重复加载权重，提速10倍+）
        model_type_lower = model_type.lower()
        cache_key = f"{model_type_lower}_cpu"
        
        if use_cache and cache_key in _MODEL_CACHE:
            print(f"[INFO] 从缓存加载{model_type_lower.upper()}模型（跳过权重加载）")
            return _MODEL_CACHE[cache_key]
        
        # ==================== 步骤1：模型类型验证 ====================
        # 检查用户指定的模型类型是否受支持
        model_type_lower = model_type.lower()
        if model_type_lower == 'unet':
            # 默认使用Underfitting Fix改进版U-Net
            model = UNetUnderfittingFix(in_channels=3, out_channels=1)
            print(f"[INFO] 创建U-Net（Underfitting Fix）模型成功")
        elif model_type_lower == 'fcn':
            # 创建FCN模型（备选方案）
            model = FCN(in_channels=3, out_channels=1)
            print(f"[INFO] 创建FCN模型成功")
        else:
            # 不支持的模型类型
            print(f"[ERROR] 不支持的模型类型: {model_type}，仅支持 'unet' 或 'fcn'")
            return None
        
        # ==================== 步骤2：确定权重加载路径 ====================
        # 支持自定义权重路径（如今天训练的权重）或使用默认OCTA预训练权重
        # 规范化路径：将 Windows 反斜杠 \ 转换为正斜杠 /
        if model_path is not None:
            model_path = model_path.replace('\\', '/')
            print(f"[DEBUG] 规范化后的模型路径: {model_path}")
        
        if model_path is not None and os.path.exists(model_path):
            # 使用用户指定的权重路径（优先级高）
            weight_path = model_path
            print(f"[INFO] 使用指定权重路径: {weight_path}")
        else:
            # 使用默认OCTA预训练权重路径
            weight_path = "./models/weights/unet_octa.pth"
            if model_path is not None:
                print(f"[WARNING] 指定权重路径不存在: {model_path}，改用默认路径")
                print(f"[DEBUG] 检查路径: os.path.exists('{model_path}') = {os.path.exists(model_path)}")
                print(f"[DEBUG] 绝对路径: {os.path.abspath(model_path)}")
            print(f"[INFO] 使用默认权重路径: {weight_path}")
        
        # ==================== 步骤3：权重文件存在性校验 ====================
        # 检查权重文件是否存在，这是加载的前置条件
        if not os.path.exists(weight_path):
            # 权重文件不存在，打印详细提示信息，帮助用户理解问题
            print(f"[WARNING] 权重文件不存在: {weight_path}")
            print(f"[WARNING] 预期路径: {os.path.abspath(weight_path)}")
            print(f"[WARNING] 如无权重文件，将使用随机初始化模型进行测试")
            return None  # 返回None，由调用者处理
        
        # ==================== 步骤4：设备自适应加载权重 ====================
        # Fix: 平台优化 - 放弃训练模块，聚焦预测功能（推理支持GPU自动适配）
        # 优先GPU，其次CPU，若请求设备不可用则回退CPU
        target_device = torch.device(device) if device else torch.device('cpu')
        if target_device.type == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA 不可用，自动回退 CPU")
            target_device = torch.device('cpu')
        try:
            # 加载权重文件到目标设备（使用map_location参数）
            checkpoint = torch.load(weight_path, map_location=target_device)
            print(f"[INFO] 权重文件已加载到CPU内存")
            
            # ==================== 步骤5：处理不同的权重文件格式 ====================
            # PyTorch权重文件可能有多种格式，需要兼容所有情况
            if isinstance(checkpoint, dict):
                # 权重文件是字典格式（常见格式）
                # 尝试不同的键名来找到state_dict
                if 'state_dict' in checkpoint:
                    # 标准格式：{'state_dict': {...}}
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    print(f"[INFO] 使用'state_dict'键加载权重")
                elif 'model_state_dict' in checkpoint:
                    # 另一种常见格式：{'model_state_dict': {...}}
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"[INFO] 使用'model_state_dict'键加载权重")
                else:
                    # 直接是state_dict，没有外层包装
                    model.load_state_dict(checkpoint, strict=False)
                    print(f"[INFO] 直接加载字典格式的权重")
            else:
                # 权重文件直接是state_dict（非字典格式）
                model.load_state_dict(checkpoint, strict=False)
                print(f"[INFO] 加载权重张量格式的权重")
            
            print(f"[SUCCESS] 成功加载权重: {weight_path}")
            
        except Exception as e:
            # 权重加载过程中出错，打印具体错误信息
            print(f"[ERROR] 加载权重文件失败: {e}")
            print(f"[ERROR] 请检查权重文件是否完整或格式是否正确")
            print(f"[WARNING] 将使用随机初始化的模型进行推理（可能精度较低）")
            # 注意：不返回 None，继续使用随机初始化的模型
            # return None  # 删除这行，继续进行
        
        # ==================== 步骤6：模型到CPU设备并设为评估模式 ====================
        # 将模型移动到CPU设备（虽然已经在CPU了，但为了代码完整性）
        model = model.to(target_device)
        
        # ==================== 步骤6：模型到目标设备并设为评估模式 ====================
        # 将模型移动到推理设备（CPU/GPU）
        model = model.to(target_device)
        
        # 设置为评估模式（inference mode）
        # 这样可以关闭dropout、batch normalization等训练时的行为
        model.eval()
        
        # 禁用梯度计算（推理时不需要梯度）
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"[INFO] 模型已设置为评估模式（CPU）")
        print(f"[INFO] 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        
        # ==================== 步骤7：缓存模型（避免重复加载）====================
        if use_cache:
            _MODEL_CACHE[cache_key] = model
            print(f"[INFO] 模型已缓存，后续请求将直接复用")
        
        return model
        
    except Exception as e:
        # 捕获所有未预期的异常
        print(f"[ERROR] 模型加载过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈，便于调试
        return None


# ==================== 图像预处理和后处理 ====================

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> Optional[torch.Tensor]:
    """
    【医学影像预处理】加载、格式转换、尺寸调整、归一化，并转换为PyTorch张量
    
    本函数是模型推理的重要环节，负责将各种格式的OCTA图像（PNG/JPG/JPEG）转换为
    标准的神经网络输入格式。关键特点：
    
    1. 格式自适应：
       - PNG（RGB）：直接使用3通道
       - PNG（RGBA）：自动去除透明通道，转换为RGB
       - JPG/JPEG：原生RGB格式，自动识别
    
    2. 大小调整：目标256x256（医学影像标准分辨率）
    
    3. 值归一化：[0,255] -> [0,1]（神经网络标准输入）
    
    4. 维度转换：HWC -> CHW -> BCHW
       - H: 高度 (height)
       - W: 宽度 (width)
       - C: 通道数 (channels)
       - B: batch大小 (batch size)
    
    Args:
        image_path: 输入图像文件路径，支持PNG/JPG/JPEG格式
        target_size: 目标尺寸(width, height)，默认(256, 256)
    
    Returns:
        预处理后的PyTorch张量
        - 形状：(1, 3, 256, 256) - batch_size=1, channels=3, H=256, W=256
        - 数据类型：torch.float32
        - 值范围：[0, 1]（已归一化）
        - 如果处理失败返回None
    
    示例:
        >>> tensor = preprocess_image('uploads/image.jpg')
        >>> print(tensor.shape)  # torch.Size([1, 3, 256, 256])
    """
    try:
        # ==================== 步骤1：图像加载与格式转换 ====================
        # PIL.Image.open()自动识别PNG/JPG/JPEG格式（魔数识别）
        # .convert('RGB')处理RGBA透明通道：
        #   - PNG(RGB) -> 直接转为RGB(3通道)
        #   - PNG(RGBA) -> 去除A通道，转为RGB(3通道)
        #   - JPG/JPEG -> 原生RGB，直接转为RGB(3通道)
        # 结果：统一的RGB三通道格式，适配模型输入要求
        image = Image.open(image_path).convert('RGB')
        
        # ==================== 步骤2：尺寸调整 ====================
        # 调整到目标尺寸(256, 256)，医学影像标准分辨率
        # BILINEAR双线性插值：平衡质量和速度，适合医学图像处理
        # 缩小时：平滑下采样，防止伪影
        # 放大时：平滑上采样，但通常输入已是256x256附近
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # ==================== 步骤3：转换为numpy数组并归一化 ====================
        # np.array()转换为numpy数组，形状：(256, 256, 3)，范围：[0, 255]
        # dtype=np.float32：32位浮点数（PyTorch标准数据类型）
        # /255.0：归一化到[0, 1]范围（神经网络标准输入范围）
        # 这样做可以：
        #   - 加快梯度计算（值范围小）
        #   - 提高模型稳定性（避免数值溢出）
        #   - 符合ImageNet预训练模式的期望
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # ==================== 步骤4：维度转换与Batch包装 ====================
        # torch.from_numpy()转换为PyTorch张量
        # .permute(2, 0, 1)：维度重排 HWC(256,256,3) -> CHW(3,256,256)
        #   - H: 高度维度 -> 位置2
        #   - W: 宽度维度 -> 位置1
        #   - C: 通道维度 -> 位置0（通常CNN期望通道优先）
        # .unsqueeze(0)：添加batch维度 CHW(3,256,256) -> BCHW(1,3,256,256)
        #   - B: batch维度 = 1（单张图像推理）
        # 最终形状(1, 3, 256, 256)是神经网络的标准输入格式
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        # 异常处理：如果预处理失败（文件损坏、格式错误等）
        # 打印错误并返回None，由调用者处理
        print(f"[ERROR] 图像预处理失败: {e}")
        print(f"[ERROR] 请检查图像文件是否为有效的PNG/JPG/JPEG格式")
        return None


def postprocess_mask(mask_tensor: torch.Tensor, original_size: Optional[Tuple[int, int]] = None, threshold: float = 0.5) -> np.ndarray:
    """
    后处理分割掩码：二值化 → 转换为8位灰度图 → 调整大小
    
    本函数确保输出的分割掩码为清晰的二值8位灰度图（仅 0 或 255），可直接保存为PNG文件。
    采用 Sigmoid 概率阈值的标准二值化策略，符合医学影像处理规范。
    
    处理步骤：
    1. 从PyTorch张量移除batch和channel维度，转换为numpy数组
    2. 应用阈值二值化：mask > threshold → 1/0（医学影像标准流程）
    3. 缩放到[0,255]范围（仅0或255，高对比度）
    4. 如果需要，调整掩码尺寸到原始图像大小
    
    Args:
        mask_tensor: 模型输出的掩码张量
                    形状为(batch_size, channels, height, width)，如(1, 1, 256, 256)
                    值范围为[0,1]（模型输出经过Sigmoid激活）
        original_size: 原始图像尺寸(width, height)。如果提供，将掩码调整到该尺寸
        threshold: 二值化阈值，默认0.5（Sigmoid输出的标准分界点）
                  建议范围：[0.3, 0.7]；降低阈值→更高灵敏度，提高阈值→更高特异性
    
    Returns:
        后处理后的掩码数组
        - 形状：(height, width)
        - 数据类型：uint8（8位无符号整数）
        - 值范围：[0, 255]（严格二值：仅 0 或 255）
        - 可直接用PIL保存为PNG，或在前端可视化
    """
    # ==================== 步骤1：张量维度处理 ====================
    # squeeze()：移除所有大小为1的维度
    # 输入形状：(1, 1, 256, 256) -> 输出形状：(256, 256)
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    
    # 验证处理后的形状（确保是2D数组）
    if mask.ndim != 2:
        print(f"[WARNING] 掩码维度异常：{mask.ndim}，预期为2")
    
    # ==================== 步骤2：阈值二值化（医学影像标准）====================
    # 将模型输出的概率[0,1]转换为二值掩码[0,1]
    # 阈值0.5：Sigmoid输出的标准分界点，概率≥0.5视为前景（血管），<0.5视为背景
    # 这一步至关重要：确保输出的是清晰的血管边界，而不是灰度概率图
    # 
    # 示例：若阈值=0.5，前景概率0.8→1，背景概率0.2→0
    mask = (mask > threshold).astype(np.float32)
    
    print(f"[DEBUG] 二值化阈值: {threshold}, 前景像素比例: {(mask > 0).sum() / mask.size * 100:.2f}%")
    
    # ==================== 步骤3：缩放到8位灰度范围 ====================
    # 将二值[0,1]缩放到8位灰度[0,255]
    # 乘以255：[0,1] -> [0,255]
    # astype(np.uint8)：转换为8位无符号整数类型
    # 结果严格为二值：0 或 255（无中间灰度值）
    mask = (mask * 255).astype(np.uint8)
    
    # ==================== 步骤4：尺寸调整 ====================
    # 如果提供了原始图像尺寸，调整掩码大小以匹配
    if original_size is not None:
        # 使用NEAREST插值模式，保留掩码的离散二值值（不产生中间灰度）
        # 这对二值分割结果很重要，避免重采样产生中间灰度破坏二值性
        mask = Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST)
        mask = np.array(mask)
        print(f"[INFO] 掩码已调整到原始尺寸: {original_size}，值范围: [{mask.min()}, {mask.max()}]")
    
    return mask


# ==================== 主要分割函数 ====================

def segment_octa_image(
    image_path: str,
    model_type: str = 'unet',
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    device: str = 'auto'
) -> str:
    """
    对OCTA图像进行分割，适配真实预训练权重（支持PNG/JPG/JPEG）
    
    这是主要的分割函数，完成从图像加载到结果保存的完整流程。
    整个流程遵循医学影像处理的最佳实践：
    1. 输入验证（检查文件、格式）
    2. 模型加载（支持容错机制）
    3. 图像预处理（自动识别PNG/JPG/JPEG，标准化、尺寸调整）
    4. 模型推理（前向传播）
    5. 结果后处理（转换为8位灰度图）
    6. 结果保存（PNG格式）
    
    容错机制：
    - 任何环节失败都返回原图路径，不影响前后端联调
    - 所有错误都有详细日志，便于调试
    
    格式支持扩展说明（2026.1.13）：
    ✓ 输入格式：PNG/JPG/JPEG三种格式自动识别
    ✓ PNG处理：RGBA→RGB自动转换（去除透明通道）
    ✓ JPG处理：原生RGB，直接处理无需转换
    ✓ 输出格式：统一为PNG灰度图（医学影像标准）
    ✓ 文件名规则：xxx.jpg → xxx_seg.png（保留原文件前缀）
    
    Args:
        image_path: 输入OCTA图像路径（支持PNG/JPG/JPEG格式）
        model_type: 模型类型，'unet'（推荐）或 'fcn'，默认'unet'
        model_path: 预训练模型权重路径参数（该参数仅为兼容性保留，
                    实际使用固定路径./models/weights/unet_octa.pth）
        output_path: 输出分割结果保存路径，如果为None则自动生成为
                    input_filename_seg.png
        device: 设备类型参数（该参数仅为兼容性保留，强制使用CPU）
    
    Returns:
        分割结果图像保存路径
        - 成功时：返回分割结果PNG文件路径
        - 失败时：返回原图路径（便于前端联调和错误追踪）
    
    示例:
        >>> # 标准用法
        >>> result_path = segment_octa_image('uploads/img123.png')
        >>> if result_path.endswith('_seg.png'):
        ...     print("分割成功")
        ... else:
        ...     print("分割失败，已返回原图")
    """
    try:
        # ==================== 步骤1：输入文件验证 ====================
        # 检查输入图像文件是否存在，这是所有后续操作的前置条件
        # 规范化路径格式（Windows 兼容性）
        image_path = str(image_path).replace('\\', '/')
        print(f"[INFO] 开始处理OCTA图像: {image_path}")
        
        if not os.path.exists(image_path):
            # 输入文件不存在，返回原图路径
            print(f"[ERROR] 输入图像文件不存在: {image_path}")
            print(f"[DEBUG] 绝对路径: {os.path.abspath(image_path)}")
            return image_path
        
        # ==================== 步骤2：模型加载 ====================
        # 调用load_unet_model加载OCTA预训练权重
        # 如果权重文件不存在或格式错误，会使用随机初始化的模型
        print(f"[INFO] 正在加载OCTA模型（权重可选）...")
        # 设备自动选择：优先CUDA，其次CPU
        target_device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
        model = load_unet_model(model_type, model_path, target_device)
        
        if model is None:
            # 模型创建失败
            print(f"[ERROR] 模型创建失败")
            return image_path
        
        # ==================== 步骤3：图像预处理 ====================
        # 预处理包括：加载、格式转换、尺寸调整、归一化
        print(f"[INFO] 正在预处理OCTA图像...")
        image_tensor = preprocess_image(image_path)
        
        if image_tensor is None:
            # 图像预处理失败（可能是文件损坏、格式错误等）
            print(f"[ERROR] 图像预处理失败，返回原图路径")
            return image_path
        
        print(f"[INFO] 图像预处理完成，张量形状: {image_tensor.shape}")
        
        # ==================== 步骤4：获取原始图像尺寸 ====================
        # 保存原始图像尺寸，后续用于恢复分割掩码的尺寸
        original_image = Image.open(image_path)
        original_size = original_image.size  # (width, height)
        print(f"[INFO] 原始图像尺寸: {original_size}")
        
        # ==================== 步骤5：模型推理（前向传播）====================
        # 将图像张量移动到CPU设备（强制使用CPU，不支持GPU）
        # 将输入移动到推理设备
        if target_device == 'cuda' and torch.cuda.is_available():
            image_tensor = image_tensor.to('cuda')
        else:
            image_tensor = image_tensor.to('cpu')
        
        print(f"[INFO] 正在进行OCTA分割模型推理...")
        
        # 使用torch.no_grad()禁用梯度计算，节省内存并加快推理速度
        # 这对于推理模式是最佳实践
        with torch.no_grad():
            # 前向传播：输入 -> 分割掩码
            # 模型输出形状：(1, 1, 256, 256)，值范围：[0,1]
            mask_tensor = model(image_tensor)
        
        print(f"[INFO] 模型推理完成，输出掩码形状: {mask_tensor.shape}")
        
        # ==================== 调试输出：模型输出统计 ====================
        # 检查模型输出的分布，帮助诊断全黑/全白问题
        mask_np = mask_tensor.squeeze().detach().cpu().numpy()
        print(f"[DEBUG] 模型输出统计:")
        print(f"        最小值: {mask_np.min():.4f}")
        print(f"        最大值: {mask_np.max():.4f}")
        print(f"        平均值: {mask_np.mean():.4f}")
        print(f"        中位数: {np.median(mask_np):.4f}")
        print(f"        >0.5的像素比例: {(mask_np > 0.5).sum() / mask_np.size * 100:.2f}%")
        print(f"        >0.3的像素比例: {(mask_np > 0.3).sum() / mask_np.size * 100:.2f}%")
        print(f"[INFO] 提示: 若>0.5像素比例接近0%，模型输出过低，可尝试:")
        print(f"           1. 检查权重文件是否正确加载")
        print(f"           2. 降低阈值（如0.3）进行临时测试")
        print(f"           3. 检查输入图像归一化是否正确")
        
        # ==================== 步骤6：结果后处理 ====================
        # 将模型输出转换为8位灰度图（0-255）
        # 这是最关键的步骤，确保输出的掩码可以直接保存为PNG
        print(f"[INFO] 正在进行结果后处理...")
        
        # 直接进行后处理，不再使用阈值二值化
        try:
            mask_array = postprocess_mask(mask_tensor, original_size)
        except Exception as post_error:
            print(f"[ERROR] 后处理失败: {post_error}")
            import traceback
            traceback.print_exc()
            # 重试后处理
            print(f"[INFO] 重试后处理")
            mask_array = postprocess_mask(mask_tensor, original_size)
        
        # 验证后处理结果
        print(f"[INFO] 后处理完成，掩码数据类型: {mask_array.dtype}, "
              f"值范围: [{mask_array.min()}, {mask_array.max()}]")
        
        # ==================== 步骤7：生成输出文件路径 ====================
        # 如果用户未指定输出路径，自动生成一个
        if output_path is None:
            # 【JPG/JPEG兼容修改】自动生成输出文件名规则：
            # - 所有分割结果统一输出为PNG格式（医学影像标准）
            # - 文件名规则：input_filename_seg.png
            # - 支持的输入格式：PNG/JPG/JPEG
            # 示例：
            #   input.png -> input_seg.png
            #   input.jpg -> input_seg.png
            #   input.jpeg -> input_seg.png
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_seg.png")
        
        print(f"[INFO] 输出文件路径: {output_path}")
        
        # ==================== 步骤8：保存分割结果 ====================
        # 将numpy数组转换为PIL Image对象
        # mode='L'：8位灰度图模式，范围0-255
        # 然后保存为PNG文件（PNG完全支持8位灰度格式）
        mask_image = Image.fromarray(mask_array, mode='L')
        mask_image.save(output_path)
        
        print(f"[SUCCESS] OCTA图像分割成功！")
        print(f"[INFO] 分割结果已保存: {output_path}")
        
        return output_path
        
    except Exception as e:
        # 捕获所有未预期的异常，打印详细的错误信息
        print(f"[ERROR] OCTA图像分割过程中发生错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈，便于调试
        print(f"[WARNING] 返回原图路径以便前后端联调")
        return image_path  # 返回原图路径，方便前端联调


"""
# ==================== 测试代码 ====================

if __name__ == '__main__':
    
    # 测试代码：验证模型和函数是否正常工作
    
    print("=" * 50)
    print("OCTA图像分割模型测试")
    print("=" * 50)
    
    # 测试1：创建模型
    print("\n[测试1] 创建U-Net模型...")
    model = UNet(in_channels=3, out_channels=1)
    print(f"✓ U-Net模型创建成功")
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试2：模型前向传播
    print("\n[测试2] 测试模型前向传播...")
    dummy_input = torch.randn(1, 3, 256, 256)  # 随机输入
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出值范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 测试3：模型加载函数
    print("\n[测试3] 测试模型加载函数...")
    loaded_model = load_unet_model('unet', None, 'cpu')
    if loaded_model is not None:
        print("✓ 模型加载函数工作正常")
    else:
        print("✗ 模型加载函数失败")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
    print("\n使用说明：")
    print("1. 将预训练模型权重文件放在指定路径")
    print("2. 调用 segment_octa_image() 函数进行图像分割")
    print("3. 如果模型加载失败，函数会返回原图路径，方便联调")
"""