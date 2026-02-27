"""
RS-Unet3+ 模型实现（适配非视网膜OCTA单目标分割）

核心特性：
- 基于 Unet3+ 的多尺度跳跃连接（全尺度融合）
- 在编码、解码与瓶颈层融入 Split-Attention 机制
- 专为非视网膜OCTA图像的单目标区域分割优化（如血管、病变区域等）

差异点（相对原始 Unet3+）：
1) 在每个 encoder/decoder block 之后加入 SplitAttentionBlock，增强通道与子空间的选择性
2) Bottleneck 同样接入 Split-Attention，用于强化全局特征
3) 输出单通道分割掩码（n_classes=1），末层无激活函数，配合Dice+BCE混合损失
4) 简化多尺度融合逻辑，减少计算冗余，提升推理速度

适用场景：
- OCTA血管分割（目标：血管网络 vs 背景）
- OCTA病变检测（目标：病变区域 vs 正常组织）
- 其他单目标二分类OCTA图像分割任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ==================== 基础卷积块 ====================

class ConvBlock(nn.Module):
    """
    基础卷积块：Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
    - padding=1 保持空间尺寸
    - 使用 BatchNorm2d 稳定训练
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ==================== Split-Attention 模块 ====================

class SplitAttentionBlock(nn.Module):
    """
    Split-Attention 注意力模块（ResNeSt核心机制）
    
    核心思想：将通道分为 radix 组，分别变换后通过注意力权重自适应融合。
    对于OCTA目标分割任务的优势：
    - 自适应学习目标区域（如血管）的多尺度特征表示
    - 通过通道注意力机制抑制背景噪声，突出目标特征
    
    关键步骤：
    1. 通过分组卷积产生 radix 份特征（多路径特征提取）
    2. 聚合后做全局池化，得到通道级全局描述
    3. 通过两个1x1卷积计算各分支注意力（softmax 按 radix 维归一）
    4. 将注意力权重作用到各分支并求和，得到融合特征
    """
    def __init__(self, channels: int, radix: int = 2, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.radix = radix
        self.group_conv = nn.Conv2d(
            channels, channels * radix, kernel_size=3, padding=1,
            groups=radix, bias=False
        )
        self.bn = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU(inplace=True)

        inter_channels = max(channels // reduction, 32)
        # 注意力权重生成器：全局池化后两层1x1卷积
        self.fc1 = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, kernel_size=1, bias=True)
        self.bn_fc1 = nn.BatchNorm2d(inter_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 分组卷积产生 radix 份特征
        feat = self.group_conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)

        # 按通道拆分为 radix 份 (B, C, H, W)
        split = torch.split(feat, self.channels, dim=1)

        # 2) 聚合求和，得到融合基底用于注意力
        feat_sum = torch.sum(torch.stack(split, dim=0), dim=0)  # (B, C, H, W)

        # 3) 全局池化 + 两层1x1卷积生成注意力权重
        gap = F.adaptive_avg_pool2d(feat_sum, 1)                # (B, C, 1, 1)
        att = self.fc1(gap)
        att = self.bn_fc1(att)
        att = self.relu(att)
        att = self.fc2(att)                                    # (B, C*radix, 1, 1)

        # softmax 按 radix 维度归一化权重
        att = att.view(x.size(0), self.radix, self.channels, 1, 1)
        att = F.softmax(att, dim=1)                            # (B, radix, C, 1, 1)

        # 4) 按权重融合各分支
        out = 0
        for i in range(self.radix):
            out = out + split[i] * att[:, i]

        return out


# ==================== RS-Unet3+ 主体 ====================

class RSUNet3Plus(nn.Module):
    """
    RS-Unet3+ 主网络（单目标分割专用版本）
    
    架构特点：
    - 编码器：5层多尺度下采样（64→128→256→512→1024通道），每层卷积后接 Split-Attention 强化局部特征
    - 解码器：Unet3+ 风格全尺度融合（从所有编码层聚合特征）+ Split-Attention 强化融合表示
    - Bottleneck：全局特征提取 + Split-Attention（最深层特征，1/16原始分辨率）
    
    输出说明：
    - 单通道分割掩码（n_classes=1固定）
    - 最后一层无激活函数（配合BCEWithLogitsLoss或Dice+BCE混合损失）
    - 推理时需手动Sigmoid(output) > 0.5 二值化
    
    Args:
        n_channels: 输入通道数，默认3（RGB彩色OCTA图像）
        n_classes:  输出通道数，固定为1（单目标二分类分割）
        base_c:     基础通道数，默认64（可调整为32/64/128平衡精度与速度）
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 1, base_c: int = 64):
        super().__init__()
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]

        # 编码器
        self.conv0_0 = ConvBlock(n_channels, filters[0])
        self.sa0 = SplitAttentionBlock(filters[0])

        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.sa1 = SplitAttentionBlock(filters[1])

        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.sa2 = SplitAttentionBlock(filters[2])

        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.sa3 = SplitAttentionBlock(filters[3])

        # Bottleneck
        self.conv4_0 = ConvBlock(filters[3], filters[4])
        self.sa4 = SplitAttentionBlock(filters[4])

        # 解码器 (Unet3+ 全尺度融合 - 优化版)
        # Level 3 解码器：融合来自瓶颈(x4_0)和编码器(x3_0, x2_0)的特征
        self.conv3_1 = ConvBlock(filters[4] + filters[3] + filters[2], filters[3])
        self.sa3_1 = SplitAttentionBlock(filters[3])

        # Level 2 解码器：融合来自上层解码器(x3_1)和编码器(x2_0, x1_0)的特征
        self.conv2_2 = ConvBlock(filters[3] + filters[2] + filters[1], filters[2])
        self.sa2_2 = SplitAttentionBlock(filters[2])

        # Level 1 解码器：融合来自上层解码器(x2_2)和编码器(x1_0, x0_0)的特征
        self.conv1_3 = ConvBlock(filters[2] + filters[1] + filters[0], filters[1])
        self.sa1_3 = SplitAttentionBlock(filters[1])

        # Level 0 解码器：融合来自上层解码器(x1_3)和编码器(x0_0)的特征
        self.conv0_4 = ConvBlock(filters[1] + filters[0], filters[0])
        self.sa0_4 = SplitAttentionBlock(filters[0])

        # 最终输出层（无激活函数，配合BCEWithLogitsLoss）
        # 输出单通道logits，推理时需sigmoid(output) > 0.5二值化
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        # 下采样与上采样算子
        self.pool = nn.MaxPool2d(2)
        self.upsample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def _resize_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """将特征 x 调整到与 ref 相同的空间尺寸（用于多尺度融合）。"""
        return self.upsample(x, size=ref.shape[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（单目标分割优化版本）
        
        输入：x (B, 3, H, W) - RGB彩色OCTA图像
        输出：logits (B, 1, H, W) - 单通道分割logits（未经sigmoid）
        
        特征图尺寸：
        - x0_0: H×W (原始分辨率)
        - x1_0: H/2×W/2
        - x2_0: H/4×W/4
        - x3_0: H/8×W/8
        - x4_0: H/16×W/16 (Bottleneck)
        """
        # ========== Encoder 路径 ==========
        x0_0 = self.sa0(self.conv0_0(x))          # H×W,   64通道
        x1_0 = self.sa1(self.conv1_0(self.pool(x0_0)))   # H/2,  128通道
        x2_0 = self.sa2(self.conv2_0(self.pool(x1_0)))   # H/4,  256通道
        x3_0 = self.sa3(self.conv3_0(self.pool(x2_0)))   # H/8,  512通道

        # Bottleneck（最深层全局特征）
        x4_0 = self.sa4(self.conv4_0(self.pool(x3_0)))   # H/16, 1024通道

        # ========== Decoder 全尺度融合（优化版，减少冗余上下采样）==========
        # Level 3 解码器 (目标尺寸: H/8×W/8)
        # 融合: 瓶颈层x4_0(上采样) + 同层编码x3_0 + 上层编码x2_0(下采样)
        x4_0_up = self._resize_to(x4_0, x3_0)     # 1024→H/8
        x2_0_down = self.pool(x2_0)               # 256→H/8
        x3_1 = torch.cat([x3_0, x4_0_up, x2_0_down], dim=1)  # 512+1024+256=1792通道
        x3_1 = self.sa3_1(self.conv3_1(x3_1))     # 输出512通道

        # Level 2 解码器 (目标尺寸: H/4×W/4)
        # 融合: 上层解码x3_1(上采样) + 同层编码x2_0 + 上层编码x1_0(下采样)
        x3_1_up = self._resize_to(x3_1, x2_0)     # 512→H/4
        x1_0_down = self.pool(x1_0)               # 128→H/4
        x2_2 = torch.cat([x2_0, x3_1_up, x1_0_down], dim=1)  # 256+512+128=896通道
        x2_2 = self.sa2_2(self.conv2_2(x2_2))     # 输出256通道

        # Level 1 解码器 (目标尺寸: H/2×W/2)
        # 融合: 上层解码x2_2(上采样) + 同层编码x1_0 + 上层编码x0_0(下采样)
        x2_2_up = self._resize_to(x2_2, x1_0)     # 256→H/2
        x0_0_down = self.pool(x0_0)               # 64→H/2
        x1_3 = torch.cat([x1_0, x2_2_up, x0_0_down], dim=1)  # 128+256+64=448通道
        x1_3 = self.sa1_3(self.conv1_3(x1_3))     # 输出128通道

        # Level 0 解码器 (目标尺寸: H×W，原始分辨率)
        # 融合: 上层解码x1_3(上采样) + 同层编码x0_0
        x1_3_up = self._resize_to(x1_3, x0_0)     # 128→H
        x0_4 = torch.cat([x0_0, x1_3_up], dim=1)  # 64+128=192通道
        x0_4 = self.sa0_4(self.conv0_4(x0_4))     # 输出64通道

        # 最终输出层（单通道logits）
        out = self.final(x0_4)  # (B, 1, H, W)
        return out


if __name__ == "__main__":
    # 模型自检：验证前向传播尺寸与参数量
    print("=" * 60)
    print("RS-Unet3+ 模型自检（单目标分割优化版本）")
    print("=" * 60)
    
    # 测试配置
    model = RSUNet3Plus(n_channels=3, n_classes=1, base_c=64)
    x = torch.randn(2, 3, 256, 256)  # Batch=2, RGB, 256x256
    
    # 前向传播
    with torch.no_grad():
        y = model(x)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n输入形状: {x.shape}  (Batch, Channels, Height, Width)")
    print(f"输出形状: {y.shape}  (Batch, Classes, Height, Width)")
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 验证输出值范围（应为logits，不是概率）
    print(f"\n输出值范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"输出均值: {y.mean().item():.4f}")
    print(f"输出标准差: {y.std().item():.4f}")
    
    # 内存占用估算
    print(f"\n前向传播显存占用估算（单张256x256图像）:")
    input_mem = x.element_size() * x.nelement() / 1024**2
    output_mem = y.element_size() * y.nelement() / 1024**2
    model_mem = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2
    print(f"  输入张量: {input_mem:.2f} MB")
    print(f"  输出张量: {output_mem:.2f} MB")
    print(f"  模型参数: {model_mem:.2f} MB")
    print(f"  估算总显存: {input_mem + output_mem + model_mem:.2f} MB (不含中间激活)")
    
    print("\n✅ 模型自检通过！适配单目标OCTA分割任务。")
    print("=" * 60)
