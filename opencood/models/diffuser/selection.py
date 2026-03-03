import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn.functional as F

from opencood.models.deTr.deCooper import DeCooper
from opencood.models.denoising_diffusion_pytorch.mask_cond_unet import Mlp


def inverse_attention_normalization(feature_map, eps=1e-6):
    """
    通过归一化和取反生成反向关注权重矩阵

    参数:
    feature_map: 输入特征图，形状为(B, C, H, W)
    eps: 防止除零的小常数

    返回:
    inverse_weights: 反转关注的权重矩阵，形状为(B, 1, H, W)
    """
    # 计算每个样本的空间注意力图 (B, 1, H, W)
    spatial_attention = torch.mean(feature_map, dim=1, keepdim=True)

    # 全局最大最小值归一化
    max_val = torch.max(spatial_attention.view(spatial_attention.size(0), -1), dim=1)[0].view(-1, 1, 1, 1)
    min_val = torch.min(spatial_attention.view(spatial_attention.size(0), -1), dim=1)[0].view(-1, 1, 1, 1)

    # 防止除零
    normalized_attention = (spatial_attention - min_val) / (max_val - min_val + eps)

    # 反转关注：用1减去归一化后的注意力值
    inverse_weights = 1.0 - normalized_attention

    # 重新归一化确保权重范围在[0,1]
    inverse_weights = (inverse_weights - inverse_weights.min()) / (inverse_weights.max() - inverse_weights.min() + eps)

    return inverse_weights


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1x1卷积生成门控权重图
        self.gate_conv = nn.Conv2d(2 * channels, 1, kernel_size=1)

    def forward(self, feat_normal, feat_high_response):
        """
        输入：
        feat_normal:    正常特征 [B, C, H, W]
        feat_high_response: 高响应特征 [B, C, H, W]
        输出：
        fused_feature: 融合特征 [B, C, H, W]
        """
        # 沿通道维度拼接特征
        concat_feat = torch.cat([feat_normal, feat_high_response], dim=1)  # [B, 2C, H, W]

        # 生成空间注意力门控图
        gate_map = torch.sigmoid(self.gate_conv(concat_feat))  # [B, 1, H, W]

        # 门控融合公式
        fused_feature = gate_map * feat_high_response + (1 - gate_map) * feat_normal

        return fused_feature

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (1) 表示输出的高度和宽度都被设置为 1。
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class DualQDeformableAttention(nn.Module):
    def __init__(
        self,
        dim,                    # 输入特征维度
        heads=8,                # 注意力头数
        n_points=4,             # 每个查询的采样点数
        qkv_bias=False,         # 是否在 QKV 投影中使用偏置
        dropout=0.1,            # Dropout 率
        stride=1,               # 采样步长
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.n_points = n_points
        self.stride = stride
        self.scale = (dim // heads) ** -0.5

        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)  # 共享 V

        # 双偏移量生成（独立分支）
        self.offset_pred_q1 = nn.Linear(dim, heads * n_points * 2)
        self.offset_pred_q2 = nn.Linear(dim, heads * n_points * 2)

        # 双注意力权重生成（独立分支）
        self.attn_weight_pred_q1 = nn.Sequential(
            nn.Linear(dim, heads * n_points),
            nn.Softmax(dim=-1)
        )
        self.attn_weight_pred_q2 = nn.Sequential(
            nn.Linear(dim, heads * n_points),
            nn.Softmax(dim=-1)
        )

        # 输出投影（融合双 Q 结果）
        self.to_out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2, x3):
        """
        Args:
            x1: (B, C, H, W) 输入特征1（生成 Q1）
            x2: (B, C, H, W) 输入特征2（生成 Q2，与 x1 互补）
            x3: (B, C, H, W) 共享的键值特征
        Returns:
            out: (B, C, H, W) 融合后的输出
        """
        B, C, H, W = x1.shape
        device = x1.device
        N = H * W
        C_head = C // self.heads

        # 将输入转换为 B,N,C 格式
        x1_flat = rearrange(x1, 'b c h w -> b (h w) c')  # (B, N, C)
        x2_flat = rearrange(x2, 'b c h w -> b (h w) c')
        x3_flat = rearrange(x3, 'b c h w -> b (h w) c')

        # --- 生成共享的 V ---
        v = self.to_v(x3_flat)
        v = rearrange(v, 'b n (h d) -> b h d n', h=self.heads)  # (B, heads, C_head, N)
        v = rearrange(v, 'b h d (p q) -> b h d p q', p=H, q=W)  # (B, heads, C_head, H, W)

        # --- 构建参考点网格 ---
        ref_x = torch.linspace(0.5, W - 0.5, W, device=device)
        ref_y = torch.linspace(0.5, H - 0.5, H, device=device)
        ref_grid = torch.stack(torch.meshgrid(ref_x, ref_y, indexing='ij'), -1).permute(1, 0, 2)  # (H, W, 2)
        ref_grid = ref_grid.reshape(1, 1, H * W, 1, 2).expand(B, self.heads, N, self.n_points, 2)  # (B, h, N, p, 2)
        ref_grid = ref_grid / torch.tensor([W, H], device=device)  # 归一化到 [0, 1]

        # --- 为 Q1 生成偏移并采样 ---
        offsets_q1 = self.offset_pred_q1(x1_flat).view(B, N, self.heads, self.n_points, 2)  # (B, N, h, p, 2)
        offsets_q1 = offsets_q1.permute(0, 2, 1, 3, 4)  # (B, h, N, p, 2)

        sample_pos_q1 = ref_grid + offsets_q1
        sample_pos_q1 = 2.0 * sample_pos_q1 - 1.0  # [-1, 1] 范围

        sampled_v_q1 = []
        for b in range(B):
            for h in range(self.heads):
                v_bh = v[b, h].unsqueeze(0)  # (1, C_head, H, W)
                pos_bh = sample_pos_q1[b, h].unsqueeze(0)  # (1, N, p, 2)
                sampled = F.grid_sample(
                    v_bh, pos_bh, mode='bilinear', padding_mode='zeros', align_corners=False
                )  # (1, C_head, N, p)
                sampled_v_q1.append(sampled)
        sampled_v_q1 = torch.cat(sampled_v_q1, dim=0).view(B, self.heads, C_head, N, self.n_points)  # (B, h, d, N, p)

        attn_weights_q1 = self.attn_weight_pred_q1(x1_flat).view(B, N, self.heads, self.n_points)  # (B, N, h, p)
        attn_weights_q1 = attn_weights_q1.permute(0, 2, 1, 3)  # (B, h, N, p)

        # 加权聚合
        out_q1 = torch.einsum('b h n p, b h d n p -> b h d n', attn_weights_q1, sampled_v_q1)
        out_q1 = rearrange(out_q1, 'b h d n -> b n (h d)')  # (B, N, C)

        # --- 同样的操作应用于 Q2 ---
        offsets_q2 = self.offset_pred_q2(x2_flat).view(B, N, self.heads, self.n_points, 2)
        offsets_q2 = offsets_q2.permute(0, 2, 1, 3, 4)

        sample_pos_q2 = ref_grid + offsets_q2
        sample_pos_q2 = 2.0 * sample_pos_q2 - 1.0

        sampled_v_q2 = []
        for b in range(B):
            for h in range(self.heads):
                v_bh = v[b, h].unsqueeze(0)
                pos_bh = sample_pos_q2[b, h].unsqueeze(0)
                sampled = F.grid_sample(
                    v_bh, pos_bh, mode='bilinear', padding_mode='zeros', align_corners=False
                )
                sampled_v_q2.append(sampled)
        sampled_v_q2 = torch.cat(sampled_v_q2, dim=0).view(B, self.heads, C_head, N, self.n_points)

        attn_weights_q2 = self.attn_weight_pred_q2(x2_flat).view(B, N, self.heads, self.n_points)
        attn_weights_q2 = attn_weights_q2.permute(0, 2, 1, 3)

        out_q2 = torch.einsum('b h n p, b h d n p -> b h d n', attn_weights_q2, sampled_v_q2)
        out_q2 = rearrange(out_q2, 'b h d n -> b n (h d)')

        # --- 融合双 Q 结果 ---
        out = torch.cat([out_q1, out_q2], dim=-1)  # (B, N, 2C)
        out = self.to_out(out)  # (B, N, C)

        # 返回原始图像格式
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out

class CollaborativeFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cbam = CBAM(in_channels)
        self.cross_att = DualQDeformableAttention(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU()
        )
        self.select =  Mlp(in_features=in_channels, out_features=1)
        self.threshold = 0.501

    def forward(self, F_self, F_others, label=None):
        # F_self: (1, C, H, W)
        # F_others: (N, C, H, W)
        B,C,H,W = F_self.shape

        # Step 1: 自车特征增强
        F1 = self.cbam(F_self)
        #print('F1',F1.shape)
        # Step 2: 生成互补区域 是否需要加个高斯中和一下
        F2 = inverse_attention_normalization(F1)
        #print('F2', F2.shape)

        Q2 = F2 * F_others # 调和一下 避免关注噪声和背景噪声
        #print('Q2', Q2.shape)

        N = F_others.size(0)
        Q1 = F1.expand(N, -1, -1, -1)
        #print('Q1', Q1.shape)

        # Step 3: 交叉注意力
        F3 = self.cross_att(Q1, Q2, F_others)   #.mean(dim=0, keepdim=True)
        #print('F3', F3.shape)

        #print(F1.shape,F3.shape,Q1.shape,Q2.shape,F_others.shape)

        # Step 4: 特征融合
        F4 = self.fusion(torch.cat([F1, F3], dim=1))

        att_map = self.select(F4)

        # # 选择前self.threshold%的token
        # flat_attention = att_map.view(B, -1)
        # # 获取 top k 值的索引
        # _, indices = torch.topk(flat_attention, k=self.threshold, dim=1, largest=True, sorted=False)
        # # 创建一个全零 mask
        # mask = torch.zeros_like(flat_attention).scatter_(1, indices, 1).view(B, 1, H, W)

        # 选择超过self.threshold的token
        confidence_map = att_map.sigmoid().max(dim=1)[0].unsqueeze(1)
        communication_mask = (confidence_map > self.threshold).float()
        communication_mask = (communication_mask - confidence_map).detach() + confidence_map
        mask = communication_mask.clamp(0, 1)
        # 特征筛选
        selected_features = F4 * mask

        return selected_features




class CollaborativeFusion_DeCooper(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cross_att = DeCooper()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU()
        )
        self.select =  Mlp(in_features=in_channels, out_features=1)
        self.threshold = 0.501

        self.top_k_ratio = 0.5

    # def forward(self, F_self, F_others):
    #     # F_self: (1, C, H, W)
    #     # F_others: (N, C, H, W)


    #     # Step 3: 交叉注意力
    #     F3_1, F3_2, F3_3 = self.cross_att(F_self, F_others)   #.mean(dim=0, keepdim=True)
    #     #print('F3', F3.shape)

    #     #print(F1.shape,F3.shape,Q1.shape,Q2.shape,F_others.shape)

    #     att_map1 = self.select(F3_1)
    #     att_map2 = self.select(F3_2)
    #     att_map3 = self.select(F3_3)

    #     # # 选择前self.threshold%的token
    #     # flat_attention = att_map.view(B, -1)
    #     # # 获取 top k 值的索引
    #     # _, indices = torch.topk(flat_attention, k=self.threshold, dim=1, largest=True, sorted=False)
    #     # # 创建一个全零 mask
    #     # mask = torch.zeros_like(flat_attention).scatter_(1, indices, 1).view(B, 1, H, W)



    #     # --- 第一个特征流 ---
    #     confidence_map1 = att_map1.sigmoid().max(dim=1)[0].unsqueeze(1)
    #     communication_mask1 = (confidence_map1 > self.threshold).float()
    #     communication_mask1 = (communication_mask1 - confidence_map1).detach() + confidence_map1
    #     mask1 = communication_mask1.clamp(0, 1)
    #     selected_features1 = F3_1 * mask1

    #     # --- 第二个特征流 ---
    #     confidence_map2 = att_map2.sigmoid().max(dim=1)[0].unsqueeze(1)
    #     communication_mask2 = (confidence_map2 > self.threshold).float()
    #     communication_mask2 = (communication_mask2 - confidence_map2).detach() + confidence_map2
    #     mask2 = communication_mask2.clamp(0, 1)
    #     selected_features2 = F3_2 * mask2

    #     # --- 第三个特征流 ---
    #     confidence_map3 = att_map3.sigmoid().max(dim=1)[0].unsqueeze(1)
    #     communication_mask3 = (confidence_map3 > self.threshold).float()
    #     communication_mask3 = (communication_mask3 - confidence_map3).detach() + confidence_map3
    #     mask3 = communication_mask3.clamp(0, 1)
    #     selected_features3 = F3_3 * mask3

    #     # ---- 添加以下代码来监控保留比例 ----
    #     total_elements = mask1.numel()
        
    #     kept_ratio1 = mask1.sum() / total_elements
    #     kept_ratio2 = mask2.sum() / total_elements
    #     kept_ratio3 = mask3.sum() / total_elements
        
    #     print(f"流1保留比例: {kept_ratio1.item():.4f} | "
    #             f"流2保留比例: {kept_ratio2.item():.4f} | "
    #             f"流3保留比例: {kept_ratio3.item():.4f}")
    #     # ------------------------------------

    #     return selected_features1, selected_features2, selected_features3

    def forward(self, F_self, F_others):
        # F_self: (1, C, H, W)
        # F_others: (N, C, H, W)

        # Step 1: 交叉注意力，生成三个不同尺度的特征图
        F3_1, F3_2, F3_3 = self.cross_att(F_self, F_others)

        # 定义一个辅助函数来执行top-k掩码操作
        def apply_topk_mask(feature_map):
            if feature_map is None:
                return None
            
            B, C, H, W = feature_map.shape
            
            # Step 2: 计算空间重要性图。
            # 通过计算通道维度的平均绝对值来评估每个像素点的重要性。
            importance_map = torch.mean(torch.abs(feature_map), dim=1) # 形状: [B, H, W]
            
            # Step 3: 计算 k 的值
            # 根据给定的比例计算需要保留的特征点数量
            num_elements = H * W
            k = int(self.top_k_ratio * num_elements)
            
            # 确保 k 至少为 1，避免全零掩码
            k = max(1, k)
            
            # Step 4: 获取 top-k 特征点的索引
            # 将重要性图展平以便排序
            flat_importance = importance_map.view(B, -1) # 形状: [B, H*W]
            _, indices = torch.topk(flat_importance, k=k, dim=1) # 获取前k个最大值的索引
            
            # Step 5: 创建并应用掩码
            # 创建一个全零掩码，并将top-k位置设置为1
            mask = torch.zeros_like(flat_importance)
            mask.scatter_(1, indices, 1.0) # 在指定索引位置填充1
            
            # 将掩码恢复到原始的空间维度 [B, 1, H, W]
            mask = mask.view(B, 1, H, W)
            
            # 将掩码应用到特征图上
            selected_features = feature_map * mask
            
            return selected_features

        # --- 为每个特征流独立应用top-k掩码 ---
        selected_features1 = apply_topk_mask(F3_1)
        selected_features2 = apply_topk_mask(F3_2)
        selected_features3 = apply_topk_mask(F3_3)

        # # ---- (可选) 监控每个流的保留比例 ----
        # def get_ratio(mask_tensor):
        #     if mask_tensor is None: return 0.0
        #     return (mask_tensor != 0).float().sum() / mask_tensor.numel()

        # ratio1 = get_ratio(selected_features1)
        # ratio2 = get_ratio(selected_features2)
        # ratio3 = get_ratio(selected_features3)
        
        # # 由于是按比例选择，打印出的比例理论上应接近 self.top_k_ratio
        # print(f"流1保留比例: {ratio1:.4f} | "
        #         f"流2保留比例: {ratio2:.4f} | "
        #         f"流3保留比例: {ratio3:.4f}")
        # # ------------------------------------

        return selected_features1, selected_features2, selected_features3