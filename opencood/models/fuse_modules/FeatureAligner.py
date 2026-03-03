import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d
import numpy as np

def get_descriptor(cls, reg):
    """提取包含分类得分的描述符"""
    # 获取分类特征中的最大得分
    scores = cls.max(dim=1)[0]  # (B, H, W)
    scores = scores.unsqueeze(1)  # (B, 1, H, W)

    # 从回归特征中提取尺寸和方向
    size = reg[:, 3:6]  # (B, 3, H, W)
    theta = reg[:, 6:7]  # (B, 1, H, W)
    ori = torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)  # (B, 2, H, W)

    # 拼接尺寸、方向和得分
    desc = torch.cat([size, ori, scores], dim=1)  # (B, 6, H, W)
    return desc.view(desc.size(0), 6, -1).permute(0, 2, 1)  # (B, H*W, 6)


# Step 2: 构建局部描述符
class LocalFeatureEncoder(nn.Module):
    # 输入维度: size(3) + orientation(2) + cls_score(1) = 6
    def __init__(self, in_dim=6, emb_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),  # LayerNorm有助于稳定训练
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# Step 3: 可微的匹配与变换估计
def differentiable_matching_and_transform(points_ego, desc_ego, points_other, desc_other):
    """
    使用"软"匹配（soft matching）和可微的SVD来估计变换
    Args:
        points_ego (B, K, 2): 自车关键点的BEV坐标 (x,y)
        desc_ego (B, K, D): 自车关键点的描述符
        points_other (B, M, 2): 它车关键点的BEV坐标 (x,y)
        desc_other (B, M, D): 它车关键点的描述符
    Returns:
        transform_matrix (B, 2, 3): 估计出的2D仿射变换矩阵
    """
    B, K, D = desc_ego.shape
    M = points_other.shape[1]

    # 1. 计算描述符相似度矩阵
    desc_ego_norm = F.normalize(desc_ego, p=2, dim=-1)
    desc_other_norm = F.normalize(desc_other, p=2, dim=-1)
    sim_matrix = torch.bmm(desc_ego_norm, desc_other_norm.transpose(1, 2))  # (B, K, M)

    # 2. "软"匹配：为每个自车点找到它车中的概率匹配分布
    # temperature参数可以控制匹配的"尖锐程度"
    temperature = 0.1
    soft_assignment = F.softmax(sim_matrix / temperature, dim=-1)  # (B, K, M)

    # 3. 计算每个自车点对应的它车中的期望点位置
    # weighted_points_other_k = sum_m (prob_k_m * point_other_m)
    expected_points_other = torch.bmm(soft_assignment, points_other)  # (B, K, 2)

    # 4. 可微的刚体变换估计 (Kabsch Algorithm implemented in PyTorch)
    # 计算质心
    centroid_ego = torch.mean(points_ego, dim=1, keepdim=True)  # (B, 1, 2)
    centroid_other = torch.mean(expected_points_other, dim=1, keepdim=True)  # (B, 1, 2)

    # 去中心化
    points_ego_centered = points_ego - centroid_ego
    points_other_centered = expected_points_other - centroid_other

    # 计算协方差矩阵 H
    H = torch.bmm(points_ego_centered.transpose(1, 2), points_other_centered)  # (B, 2, 2)

    # SVD分解
    try:
        # torch.svd 在某些版本中被弃用
        U, S, V = torch.svd(H)
    except AttributeError:
        # 新版PyTorch使用 torch.linalg.svd
        U, S, Vh = torch.linalg.svd(H)
        V = Vh.mH  # .mH 是共轭转置，对于实数矩阵就是转置

    # 计算旋转矩阵 R
    R = torch.bmm(V, U.transpose(1, 2))  # (B, 2, 2)

    # 修正反射情况（重要！）
    det = torch.det(R)  # (B)
    det_fix = torch.eye(2, device=R.device).unsqueeze(0).repeat(B, 1, 1)
    det_fix[:, 1, 1] = det.sign()
    R = torch.bmm(V, torch.bmm(det_fix, U.transpose(1, 2)))

    # 计算平移向量 t
    t = centroid_ego.transpose(1, 2) - torch.bmm(R, centroid_other.transpose(1, 2))  # (B, 2, 1)

    # 组合成 (B, 2, 3) 的仿射变换矩阵
    transform_matrix = torch.cat([R, t], dim=2)

    return transform_matrix


# 🔚 整合所有步骤的最终模块
class DifferentiableFeatureAligner(nn.Module):
    def __init__(self, num_keypoints=256, emb_dim=64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.encoder = LocalFeatureEncoder(in_dim=6, emb_dim=emb_dim)  # dairv2x 64?128?

    def forward(self, ego_cls, ego_reg, other_cls, other_reg):
        """
        输入:
        ego_cls, other_cls: (B, num_classes, H, W) e.g., (B, 2, H, W)
        ego_reg, other_reg: (B, 7*2, H, W)

        返回:
        aligned_other_cls: (B, num_classes, H, W)
        aligned_other_reg: (B, 7*2, H, W)
        """
        B, num_classes, H, W = ego_cls.shape

        B, _, H, W = ego_cls.shape
        device = ego_cls.device

        # 为关键点构建描述符
        # -- 自车 --
        ego_desc = get_descriptor(ego_cls, ego_reg)  # (B, H*W, 6)
        ego_desc = self.encoder(ego_desc)  # (B, H*W, 128)
        # -- 它车 --
        other_desc = get_descriptor(other_cls, other_reg)  # (B, H*W, 6)
        other_desc = self.encoder(other_desc)  # (B, H*W, 128)

        # Step 3: 分块计算注意力权重
        block_size = 1000  # 每次处理 1000 个点，可根据 GPU 内存调整
        num_points = H * W
        num_blocks = (num_points + block_size - 1) // block_size
        aligned_cls_flat_blocks = []
        aligned_reg_flat_blocks = []

        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, num_points)
            # 分块计算注意力
            attn_block = torch.bmm(ego_desc[:, start:end], other_desc.transpose(1, 2))  # (B, block_size, H*W)
            attn_weights_block = F.softmax(attn_block / 0.1, dim=-1)  # (B, block_size, H*W)

            # Step 4: 展平 cls 和 reg 特征（只需要做一次）
            if i == 0:
                other_cls_flat = other_cls.view(B, num_classes, -1).permute(0, 2, 1)  # (B, H*W, num_classes)
                other_reg_flat = other_reg.view(B, 14, -1).permute(0, 2, 1)  # (B, H*W, 14)

            # Step 5: 加权融合
            aligned_cls_block = torch.bmm(attn_weights_block, other_cls_flat)  # (B, block_size, num_classes)
            aligned_reg_block = torch.bmm(attn_weights_block, other_reg_flat)  # (B, block_size, 14)
            aligned_cls_flat_blocks.append(aligned_cls_block)
            aligned_reg_flat_blocks.append(aligned_reg_block)

        # 拼接所有分块结果
        aligned_cls_flat = torch.cat(aligned_cls_flat_blocks, dim=1)  # (B, H*W, num_classes)
        aligned_reg_flat = torch.cat(aligned_reg_flat_blocks, dim=1)  # (B, H*W, 14)

        # Step 6: 恢复为特征图格式
        aligned_other_cls = aligned_cls_flat.permute(0, 2, 1).view(B, num_classes, H, W)
        aligned_other_reg = aligned_reg_flat.permute(0, 2, 1).view(B, 14, H, W)

        return aligned_other_cls, aligned_other_reg


class DifferentiableFeatureAligner2(nn.Module):
    def __init__(self, num_keypoints=256, emb_dim=64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sqrt_dim = np.sqrt(64)
        self.encoder = LocalFeatureEncoder(in_dim=6, emb_dim=64)

    def forward(self, ego_cls, ego_reg, other_cls, other_reg):
        """
        输入:
        ego_cls, other_cls: (B, num_classes, H, W) e.g., (B, 2, H, W)
        ego_reg, other_reg: (B, 7*2, H, W)

        返回:
        aligned_other_cls: (B, num_classes, H, W)
        aligned_other_reg: (B, 7*2, H, W)
        """
        B, num_classes, H, W = ego_cls.shape

        B, _, H, W = ego_cls.shape
        device = ego_cls.device

        # 为关键点构建描述符
        # -- 自车 --
        ego_desc = get_descriptor(ego_cls, ego_reg)  # (1, H*W, 6)
        ego_desc = self.encoder(ego_desc)  # (1, H*W, 128)
        # -- 它车 --
        other_desc = get_descriptor(other_cls, other_reg)  # (1, H*W, 6)
        other_desc = self.encoder(other_desc)  # (1, H*W, 128)

        ego_desc = ego_desc.transpose(0, 1)  # (H*W, 1, d)
        other_desc = other_desc.transpose(0, 1).transpose(1, 2)  # (H*W, 1, d)
        attn_weights = torch.matmul(ego_desc, other_desc) / self.sqrt_dim  # (H*W, 1, 1)

        N, _, H, W = other_cls.shape
        other_cls = other_cls.permute(2, 3, 0, 1).reshape(H * W, N, -1)
        aligned_cls_flat = torch.bmm(attn_weights, other_cls)  # (H*W, 1, C) x (H*W, 1, 1) -> (H*W, 1, C)
        aligned_cls_flat = aligned_cls_flat.permute(1, 2, 0)

        other_reg = other_reg.permute(2, 3, 0, 1).reshape(H * W, N, -1)
        aligned_reg_flat = torch.bmm(attn_weights, other_reg)  # (H*W, 1, C) x (H*W, 1, 1) -> (H*W, 1, C)
        aligned_reg_flat = aligned_reg_flat.permute(1, 2, 0)

        # Step 6: 恢复为特征图格式
        aligned_other_cls = aligned_cls_flat.view(B, num_classes, H, W)
        aligned_other_reg = aligned_reg_flat.view(B, 14, H, W)

        return aligned_other_cls, aligned_other_reg


# 纠正分支：利用自车特征引导它车特征的调整
class CorrectionBranch(nn.Module):
    def __init__(self, ch_cls, ch_reg, emb_dim=128):
        super().__init__()
        # 可变形卷积，用于调整它车特征
        self.deform_conv_cls = DeformConv2d(ch_cls, ch_cls, kernel_size=3, padding=1)
        self.deform_conv_reg = DeformConv2d(ch_reg, ch_reg, kernel_size=3, padding=1)

        # 根据自车和它车特征学习偏移量
        self.offset_conv = nn.Conv2d(emb_dim, 2 * 3 * 3, kernel_size=3, padding=1)  # 2表示x和y偏移
        # 嵌入层，用于融合自车和它车特征
        self.embed = nn.Conv2d((ch_cls + ch_reg) * 2, emb_dim, kernel_size=1)

    def forward(self, ego_cls, ego_reg, other_cls, other_reg):
        # 融合自车和它车特征以生成偏移量
        combined_features = torch.cat([ego_cls, ego_reg, other_cls, other_reg], dim=1)

        embed_feat = self.embed(combined_features)
        offset = self.offset_conv(embed_feat)
        # 使用可变形卷积调整它车特征
        corrected_other_feat_cls = self.deform_conv_cls(other_cls, offset)
        corrected_other_feat_reg = self.deform_conv_reg(other_reg, offset)

        return corrected_other_feat_cls, corrected_other_feat_reg


# 鲁棒特征对齐器：结合仿射变换和纠正分支
class RobustFeatureAligner(nn.Module):
    def __init__(self, num_keypoints=256, emb_dim=64, ch_cls=2, ch_reg=14):
        super().__init__()
        self.aligner2 = DifferentiableFeatureAligner2(num_keypoints, emb_dim)
        self.correction_branch = CorrectionBranch(ch_cls, ch_reg, emb_dim)
        # 融合层：将仿射变换结果和纠正结果融合
        self.fusion_conv_cls = nn.Conv2d((ch_cls) * 2, ch_cls, kernel_size=1)
        self.fusion_conv_reg = nn.Conv2d((ch_reg) * 2, ch_reg, kernel_size=1)

    def forward(self, ego_cls, ego_reg, other_cls, other_reg):
        # 步骤1：获取仿射变换后的特征
        aligned_other_cls, aligned_other_reg = self.aligner2(ego_cls, ego_reg, other_cls, other_reg)

        # 步骤2：纠正分支处理分类特征（回归特征类似）
        corrected_other_cls, corrected_other_reg = self.correction_branch(ego_cls, ego_reg, other_cls, other_reg)

        # 步骤3：融合仿射变换特征和纠正特征
        fused_cls = self.fusion_conv_cls(torch.cat([aligned_other_cls, corrected_other_cls], dim=1))
        fused_reg = self.fusion_conv_reg(torch.cat([aligned_other_reg, corrected_other_reg], dim=1))

        return fused_cls, fused_reg


### 如何使用
# 参数设置
num_keypoints = 512
emb_dim = 128
ch_cls = 2  # 分类通道数 (前景/背景)
ch_reg = 14  # 回归通道数 (7维 × 2)

# 创建模型实例
model = RobustFeatureAligner(
    num_keypoints=num_keypoints,
    emb_dim=emb_dim,
    ch_cls=ch_cls,
    ch_reg=ch_reg
)

import torch

B, H, W = 2, 200, 176  # batch size, height, width

# 示例输入（自车与他车）
ego_cls = torch.randn(B, ch_cls, H, W)
ego_reg = torch.randn(B, ch_reg, H, W)
other_cls = torch.randn(B, ch_cls, H, W)
other_reg = torch.randn(B, ch_reg, H, W)

# 前向传播
fused_cls, fused_reg = model(ego_cls, ego_reg, other_cls, other_reg)

print("Fused cls shape:", fused_cls.shape)  # (B, 2, H, W)
print("Fused reg shape:", fused_reg.shape)  # (B, 14, H, W)