import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.fuse_modules.STN import SpatialTransformer
from opencood.models.sub_modules.VSSD import VMAMBA2Block


def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


class AttFusion_late(nn.Module):
    def __init__(self, feature_dim, H=None, W=None, chunk_size=None):
        super(AttFusion_late, self).__init__()

        self.att = ScaledDotProductAttention(feature_dim)

        # self.enhance = ScaledDotProductAttention(feature_dim)
        # self.att = VMAMBA2Block(feature_dim,input_resolution=[H,W],ssd_chunk_size=chunk_size)

    def forward(self, x):
        B, C, H, W = x.size()

        cav_num = x.shape[0]
        xx = x.view(cav_num, C, -1).permute(2,0,1)
        h = self.att(xx)
        h = h.permute(1,2,0).view(cav_num, C, W, H)  # N1, C, W, H

        # cav_num = x.shape[0]
        # xx = x.view(cav_num, C, -1).permute(2,0,1)
        # h = self.enhance(xx)
        # h = h.permute(1,2,0).view(cav_num, C, W, H)  # N1, C, W, H
        # h = h.view(cav_num, C, -1).permute(0,2,1)
        # #print('x before att',x.shape)
        # h = self.att(h, H, W)
        # h = h.permute(0, 2, 1).view(cav_num, C, H, W)


        return h


class Com_AttFusion(nn.Module):
    def __init__(self, feature_dims,  H=None, W=None, chunk_size=None):
        super(Com_AttFusion, self).__init__()

        #self.att = ScaledDotProductAttention(feature_dims)

        # mamba
        self.att = VMAMBA2Block(feature_dims,input_resolution=[H,W],ssd_chunk_size=chunk_size)

        #linear
        #self.att = LinearAttention(feature_dims)

        self.competive = CompetitiveAttentionFusion(in_channels=feature_dims, )

    def forward(self, xx, record_len, mask=None):
        """
        Fusion forwarding.

        Parameters
        ----------
        xx : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)

        record_len : list
            shape: (B)

        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego,
            shape: (B, L, L, 2, 3)

        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = xx.shape

        split_x = regroup(xx, record_len)
        out = []
        # iterate each batch
        for xx in split_x:
            i = 0  # ego
            x = xx
            cav_num = x.shape[0]
            # print('x',x.shape)
            tensor_list = list(x.split(1, dim=0))
            x = self.competive(tensor_list)

            # # att
            # # print('x', x.shape)
            # x = x.view(1, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel.
            # h = self.att(x)
            # h = h.permute(1, 2, 0).view(1, C, H, W)  # C, W, H before

            # mamba
            x = x.view(1, C, -1).permute(0,2,1)
            #print('x before att',x.shape)
            h = self.att(x, H, W)
            h = h.permute(0, 2, 1).view(1, C, H, W)
            #print('x after att',x.shape)


            out.append(h[0, ...])

        out = torch.stack(out)
        return out


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim
        dim (int): dimension of attention
    Inputs: x
        - **x** (batch, seq_len, d_model): tensor containing the input sequence
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.query_linear = nn.Linear(dim, dim)
        self.key_linear = nn.Linear(dim, dim)
        self.value_linear = nn.Linear(dim, dim)

    def forward(self, x):
        # Generate query, key, and value from the input x
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        # Compute the attention scores
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        # Apply softmax to get attention weights
        attn = F.softmax(score, -1)

        # Compute the context vector
        context = torch.bmm(attn, value)

        return context


class LinearAttention(nn.Module):
    """
    Linear Attention with O(N) complexity
    Inputs: query, key, value
        - query: (B, N, C)
        - key:   (B, M, C)
        - value: (B, M, C)
    Returns:
        - context: (B, N, C)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.scale = dim ** -0.5  # 或者根据需要调整

    def forward(self, query, key, value):
        B, N, C = query.shape
        M = key.shape[1]

        # 先归一化 Q 和 K（可选）
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        # 计算 K^T @ V -> (B, C, C)
        key_value = torch.bmm(key.transpose(1, 2), value)  # (B, C, C)

        # 计算 Q @ (K^T @ V)
        context = torch.bmm(query, key_value)  # (B, N, C)
        context = context / N  # 可选归一化

        return context


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        """
        多头空间自注意力模块：多视角建模、更强的空间依赖建模能力，复杂任务

        参数:
            in_channels (int): 输入特征图的通道数
            num_heads (int): 注意力头的数量
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert self.head_dim * num_heads == in_channels, "in_channels 必须能被 num_heads 整除"

        # Query/Key/Value 投影
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 输出投影
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 可学习权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 输入特征图，形状为 (B, C, H, W)

        返回:
            Tensor: 自注意力增强后的特征图，形状为 (B, C, H, W)
        """
        B, C, H, W = x.size()

        # 投影到 Query, Key, Value
        Q = self.query(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # (B, H, HW, D)
        K = self.key(x).view(B, self.num_heads, self.head_dim, H * W)  # (B, H, D, HW)
        V = self.value(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # (B, H, HW, D)

        # 计算注意力权重
        energy = torch.matmul(Q, K) / (self.head_dim ** 0.5)  # (B, H, HW, HW)
        attention = F.softmax(energy, dim=-1)

        # 应用注意力
        out = torch.matmul(attention, V)  # (B, H, HW, D)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)  # (B, C, H, W)

        # 输出投影 + 残差连接
        out = self.gamma * self.out_proj(out) + x

        return out

#传输一次
# class CompetitiveAttentionFusion(nn.Module):
#     def __init__(self, in_channels, hidden_dim=64, num_heads=8):
#         """
#         竞争注意力融合模块 + 多头自注意力机制

#         参数:
#             in_channels (int): 输入特征图的通道数
#             hidden_dim (int): MLP中隐藏层的通道数
#             num_heads (int): 自注意力头的数量
#         """
#         super(CompetitiveAttentionFusion, self).__init__()

#         # MLP网络用于计算每个特征图的得分图
#         self.score_mlp = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, 1, kernel_size=1)
#         )

#         # 多头自注意力模块
#         self.attention = ScaledDotProductAttention(in_channels)

#     def forward(self, features):
#         """
#         前向传播

#         参数:
#             features (List[Tensor]): 输入的特征图列表，每个形状为 (B, C, H, W)

#         返回:
#             Tensor: 融合并增强后的特征图，形状为 (B, C, H, W)
#         """
#         # Step 1: 为每个特征图生成得分图
#         scores = [self.score_mlp(feat) for feat in features]

#         # Step 2: 合并得分图，得到 (B, n, H, W)
#         scores_stack = torch.cat(scores, dim=1)

#         # Step 3: 获取每个空间位置的最佳特征图索引 (B, H, W)
#         best_indices = torch.argmax(scores_stack, dim=1)

#         # Step 4: 堆叠所有特征图 (B, n, C, H, W)
#         features_stack = torch.stack(features, dim=1)

#         # Step 5: 扩展索引以匹配通道维度
#         B,C,H,W = features[0].shape
#         best_indices_expanded = best_indices.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
#         best_indices_expanded = best_indices_expanded.expand(-1, -1, C, -1, -1)  # (B, 1, C, H, W)

#         # Step 6: 使用 gather 选择最佳特征图
#         selected_features = torch.gather(features_stack, dim=1, index=best_indices_expanded)
#         fused = selected_features.squeeze(1)  # (B, C, H, W)

#         fused = fused.view(1, C, -1).permute(2, 0, 1)
#         # Step 7: 应用多头自注意力机制
#         out = self.attention(fused)
#         out = out.permute(1, 2, 0).reshape(B, C, H, W)
#         return out



import matplotlib.pyplot as plt
import os

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def show_heatmaps(matrices, path=None, figsize=(5, 5), cmap='Blues'):
    num_rows, num_cols = matrices.shape[:2]  # 修改以适应 (B, 1, H, W) 或 (B, C, H, W)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap)
            ax.axis('off')  # 去除坐标轴

    plt.subplots_adjust(wspace=0, hspace=0) # 可选：去除子图之间的间距

    if path:
        plt.savefig(path, dpi=1300, bbox_inches='tight', pad_inches=0) # 去除白边
    plt.close(fig) # 关闭图形以释放内存





# # 传输两次
class CompetitiveAttentionFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_heads=8):
        super(CompetitiveAttentionFusion, self).__init__()
        self.score_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        self.attention = ScaledDotProductAttention(in_channels)

        # 定义一个计数器，确保每次保存的图片文件名唯一
        self.plot_counter = 0
        # 定义保存热力图的文件夹名称
        self.heatmap_output_dir = "compare"
        # 创建文件夹（如果它不存在的话）
        os.makedirs(self.heatmap_output_dir, exist_ok=True)



    def forward(self, features):
        
 # Step 1 & 2: 保持不变，计算得分图并堆叠
        scores = [self.score_mlp(feat) for feat in features]
        scores_stack = torch.cat(scores, dim=1) # (B, n, H, W)

        # ==================== 修改开始 ====================

        # Step 3 (修改): 获取每个空间位置得分最高的前2个特征图的索引和得分
        # 我们需要确保 k 不大于特征图的数量 n
        n = len(features)
        k =  min(1, n)
        topk_scores, topk_indices = torch.topk(scores_stack, k=k, dim=1)
        # topk_scores 的形状: (B, k, H, W)
        # topk_indices 的形状: (B, k, H, W)

        # Step 4: 保持不变，堆叠所有特征图
        features_stack = torch.stack(features, dim=1) # (B, n, C, H, W)
        B, C, H, W = features[0].shape

        # Step 5 & 6 (修改): 使用 gather 一次性选择出最优和次优的特征图
        # 扩展索引以匹配特征图的维度 (B, k, C, H, W)
        topk_indices_expanded = topk_indices.unsqueeze(2).expand(-1, -1, C, -1, -1)

        # 使用 gather 获取 top-k 特征
        # topk_features 的形状: (B, k, C, H, W)
        topk_features = torch.gather(features_stack, dim=1, index=topk_indices_expanded)

        # Step 6.5 (新增): 融合最优和次优特征 (推荐使用加权融合)
        # 我们使用 topk 的得分作为权重，但最好先通过 softmax 使其归一化
        weights = F.softmax(topk_scores, dim=1) # (B, k, H, W)

        # 将权重扩展到可以与特征相乘的维度 (B, k, 1, H, W) -> (B, k, C, H, W)
        weights_expanded = weights.unsqueeze(2) # .expand(-1, -1, C, -1, -1) # expand是可选的，广播机制会自动处理

        # 执行加权求和
        # (B, k, C, H, W) * (B, k, C, H, W) -> sum over k -> (B, C, H, W)
        fused = torch.sum(topk_features * weights_expanded, dim=1)

        # ==================== 修改结束 ====================

        # Step 7: 应用多头自注意力机制 (保持不变)
        fused_reshaped = fused.view(B, C, H * W).permute(2, 0, 1)
        out = self.attention(fused_reshaped)
        out = out.permute(1, 2, 0).reshape(B, C, H, W)

        return out

    # def forward(self, features):
    #     # ... (前面的步骤保持不变) ...
    #     scores = [self.score_mlp(feat) for feat in features]
    #     scores_stack = torch.cat(scores, dim=1)

    #     n = len(features)
    #     k = min(1, n)
    #     topk_scores, topk_indices = torch.topk(scores_stack, k=k, dim=1)

    #     features_stack = torch.stack(features, dim=1)
    #     B, C, H, W = features[0].shape

    #     topk_indices_expanded = topk_indices.unsqueeze(2).expand(-1, -1, C, -1, -1)
    #     topk_features = torch.gather(features_stack, dim=1, index=topk_indices_expanded) # (B, 1, C, H, W)
        
    #     # 计算加权后的 fused，但暂时不用它来可视化
    #     weights = F.softmax(topk_scores, dim=1)
    #     weights_expanded = weights.unsqueeze(2)
    #     fused = torch.sum(topk_features * weights_expanded, dim=1)

    #     # # # ==================================================================
    #     # # #             ↓↓↓ 全面诊断可视化代码 (一批次输出) ↓↓↓
    #     # # # ==================================================================
    #     if not self.training:
    #         # --- 1. 可视化拼接后的特征图 (Gathered Feature), 而非加权后 ---  <- 这里是修改点
    #         # topk_features 的形状是 (B, 1, C, H, W)，先去掉 k=1 的维度
    #         gathered_feature_map = topk_features.squeeze(1) # 形状变为 (B, C, H, W)
            
    #         # 沿着通道 C 取最大值，得到一张 2D 激活图
    #         gathered_activation_map, _ = torch.max(gathered_feature_map, dim=1, keepdim=True) # (B, 1, H, W)
            
    #         # 保存热力图
    #         gathered_heatmap_numpy = gathered_activation_map.detach().cpu().numpy()
    #         gathered_filepath = os.path.join(self.heatmap_output_dir, f"{self.plot_counter}_gathered_final.png") # 文件名改为 gathered
    #         print(f"正在保存拼接后(Gathered)的热力图至: {gathered_filepath}")
    #         show_heatmaps(gathered_heatmap_numpy, path=gathered_filepath, cmap='viridis')

    #         # --- 2. 循环可视化 n 辆车的各自情况 (保持不变) ---
    #         for i in range(n):
    #             # --- a. 可视化原始特征 (Original Feature) ---
    #             original_feat_map = features[i]
    #             original_activation, _ = torch.max(original_feat_map, dim=1, keepdim=True)
                
    #             original_heatmap_numpy = original_activation.detach().cpu().numpy()
    #             original_filepath = os.path.join(self.heatmap_output_dir, f"{self.plot_counter}_vehicle_{i}_original.png")
    #             print(f"正在保存车辆 {i} 的原始热力图至: {original_filepath}")
    #             show_heatmaps(original_heatmap_numpy, path=original_filepath, cmap='viridis')

    #             # --- b. 可视化被选择的特征 (Selected Part) ---
    #             selection_mask = (topk_indices == i).float()
    #             selected_part_activation = original_activation * selection_mask

    #             selected_part_numpy = selected_part_activation.detach().cpu().numpy()
    #             selected_part_filepath = os.path.join(self.heatmap_output_dir, f"{self.plot_counter}_vehicle_{i}_selected_part.png")
    #             print(f"正在保存车辆 {i} 被选择的热力图至: {selected_part_filepath}")
    #             show_heatmaps(selected_part_numpy, path=selected_part_filepath, cmap='viridis')

    #         # --- 3. 所有可视化完成后，递增计数器 (保持不变) ---
    #         self.plot_counter += 1

    #     # # # ==================================================================
    #     # # #                   ↑↑↑ 可视化代码结束 ↑↑↑
    #     # # # ==================================================================

    #     # Step 7: 应用多头自注意力机制 (仍然使用加权后的fused)
    #     fused_reshaped = fused.view(B, C, H * W).permute(2, 0, 1)
    #     out = self.attention(fused_reshaped)
    #     out = out.permute(1, 2, 0).reshape(B, C, H, W)

    #     return out



# # 示例：输入两个特征图，每个形状为 (B, C, H, W)
# model = CompetitiveAttentionFusion(in_channels=64, num_heads=8)
# features = [torch.randn(2, 64, 48, 176), torch.randn(2, 64, 48, 176)]

# # 前向传播
# output = model(features)

# # 输出形状应为 (B, 64, 32, 32)
# print(output.shape)  # 输出: torch.Size([2, 64, 32, 32])


