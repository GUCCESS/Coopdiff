import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from opencood.models.fuse_modules.STN import SpatialTransformer
from opencood.models.sub_modules.VSSD import VMAMBA2Block




import os
import matplotlib.pyplot as plt
import numpy as np
def get_unique_filename(base_path):
    filename, ext = os.path.splitext(base_path)
    counter = 0
    new_path = base_path
    while os.path.exists(new_path):
        new_path = f"{filename}_{counter}{ext}"
        counter += 1
    return new_path
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




class AttentionWindow(nn.Module):
    """
    Final version of the Attention Module with a strict NCHW -> NCHW interface.

    Handles all preprocessing (batching, padding, mask generation) and
    postprocessing (slicing, unbatching) internally.
    """
    def __init__(
        self,
        dim=256,
        heads=4,
        spatial_window_size=4,
        max_agent_size=5,
        dropout=0.
    ):
        super().__init__()

        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        if isinstance(spatial_window_size, int):
            self.spatial_window_size = (spatial_window_size, spatial_window_size)
        else:
            self.spatial_window_size = spatial_window_size
            
        self.max_agent_size = max_agent_size

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        max_win_d = self.max_agent_size
        max_win_h, max_win_w = self.spatial_window_size
        
        self.relative_position_bias_table = nn.Embedding(
            (2 * max_win_d - 1) * (2 * max_win_h - 1) * (2 * max_win_w - 1),
            self.heads
        )

    def _calculate_relative_position_index(self, N, H_win, W_win, device):
        coords_d = torch.arange(N, device=device)
        coords_h = torch.arange(H_win, device=device)
        coords_w = torch.arange(W_win, device=device)
        
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        max_win_d, (max_win_h, max_win_w) = self.max_agent_size, self.spatial_window_size
        
        relative_coords[:, :, 0] += max_win_d - 1
        relative_coords[:, :, 1] += max_win_h - 1
        relative_coords[:, :, 2] += max_win_w - 1
        relative_coords[:, :, 0] *= (2 * max_win_h - 1) * (2 * max_win_w - 1)
        relative_coords[:, :, 1] *= (2 * max_win_w - 1)
        return relative_coords.sum(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape (N, C, H, W) with a variable N.
        """
        # 1. --- 预处理 (Preprocessing) ---
        assert x.dim() == 4, f"Input must be NCHW format, but got {x.dim()} dimensions."
        original_n, C, H, W = x.shape
        
        assert original_n <= self.max_agent_size, \
            f"Input agent size N={original_n} exceeds model's max_agent_size={self.max_agent_size}"

        # 1a. 添加 Batch 维度 (NCHW -> 1NCHW)
        x = x.unsqueeze(0) # Shape: (1, original_n, C, H, W)

        # 1b. 创建掩码 (Mask Generation)
        # 掩码表示有效数据的位置
        mask = torch.ones(1, original_n, dtype=torch.bool, device=x.device)

        # 1c. 填充 (Padding)
        num_padding = self.max_agent_size - original_n
        if num_padding > 0:
            # F.pad 的参数格式是 (pad_last_dim, pad_dim_-1, ...)
            # 我们要填充第1维 (N)，所以是第4个位置
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, num_padding), 'constant', 0)
            
            # 填充 mask
            pad_mask = torch.zeros(1, num_padding, dtype=torch.bool, device=x.device)
            mask = torch.cat([mask, pad_mask], dim=1)
        
        # 预处理后, x 的形状为 (1, max_agent_size, C, H, W)
        # mask 的形状为 (1, max_agent_size)
        
        # 2. --- 核心注意力计算 (Core Attention Logic) ---
        B, N, C, H, W = x.shape # B=1, N=max_agent_size
        win_h, win_w = self.spatial_window_size
        x = rearrange(x, 'b n c h w -> b n h w c')
        x = rearrange(x, 'b n (h p1) (w p2) c -> (b h w) (n p1 p2) c', p1=win_h, p2=win_w)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        
        relative_position_index = self._calculate_relative_position_index(N, win_h, win_w, x.device)
        bias = self.relative_position_bias_table(relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # 应用我们之前生成的 mask
        mask = rearrange(mask, 'b n -> b n 1 1')
        window_mask = repeat(mask, 'b n 1 1 -> b (n p1 p2)', p1=win_h, p2=win_w)
        num_spatial_windows = (H // win_h) * (W // win_w)
        window_mask = repeat(window_mask, 'b s -> (b m) 1 s', m=num_spatial_windows)
        attn_mask = window_mask[:, :, :, None] * window_mask[:, :, None, :]
        sim = sim.masked_fill(attn_mask == 0, -torch.finfo(sim.dtype).max)

        attn = self.attend(sim)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (n p1 p2) c -> b n (h p1) (w p2) c',
                        b=B, h=(H // win_h), w=(W // win_w), p1=win_h, p2=win_w, n=N)
        out = rearrange(out, 'b n h w c -> b n c h w')
        
        # 计算后, out 的形状为 (1, max_agent_size, C, H, W)

        # 3. --- 后处理 (Postprocessing) ---
        # 3a. 切片 (Slicing) - 截取原始 agent 的数量
        out = out[:, :original_n, :, :, :] # Shape: (1, original_n, C, H, W)
        
        # 3b. 移除 Batch 维度 (1NCHW -> NCHW)
        out = out.squeeze(0) # Shape: (original_n, C, H, W)

        return out


class ChannelSelection(nn.Module):
    def __init__(self, feature_dims,  H=None, W=None):
        super(ChannelSelection, self).__init__()

        self.channelselect = CollaborativeFusionModule(C_orig=feature_dims, C_out=feature_dims, H=H, W=W, patch_size=1, 
                                                      score_fusion_weights=[0.6, 0.4], proportion=0.2)

        self.channelfuse = ChannelFusion(feature_dims)

        self.attention = ScaledDotProductAttention(feature_dims)
        # self.attentionwindow = AttentionWindow(dim=feature_dims, heads=4, spatial_window_size=2, max_agent_size=5, dropout=0.)


    def forward(self, features, confidence_maps=None):
        _ ,C, H, W = features[0].shape
        cav_num = len(features)
        
        fused = self.channelselect(features, confidence_maps)


        # # =================================================================
        # # =========== START: ADDED CODE FOR VISUALIZATION =================
        # # =================================================================
        # # Check if the fused tensor is not empty and has the expected dimensions
        # if fused is not None and fused.dim() == 4:
        #     # Select the first item in the batch to visualize
        #     feature_to_visualize = fused[1]
            
        #     # Take the max value across the channel dimension
        #     feature_map = torch.max(feature_to_visualize, dim=0)[0]
        #     # feature_map = torch.mean(feature_to_visualize, dim=0)
            
        #     # Move to CPU and convert to NumPy array
        #     feature_map_np = feature_map.detach().cpu().numpy()
            
        #     # Reshape for the show_heatmaps function (expects 4D array)
        #     map_to_plot = feature_map_np[np.newaxis, np.newaxis, :, :]
            
        #     # Define a NEW save directory to avoid overwriting previous visualizations
        #     save_dir = './whispernet_01_max'
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     # Get a unique filename and save the plot
        #     base_path = os.path.join(save_dir, 'fused_feature_map.png')
        #     unique_path = get_unique_filename(base_path)
        #     show_heatmaps(matrices=map_to_plot, path=unique_path, cmap='viridis')
        # # =================================================================
        # # ============ END: ADDED CODE FOR VISUALIZATION ==================
        # # =================================================================



        fused = self.channelfuse(fused)
        # out =fused
        cav_num = fused.shape[0]

        # scaledatt
        fused_reshaped = fused.view(cav_num, C, H * W).permute(2, 0, 1)
        out = self.attention(fused_reshaped)
        out = out.permute(1, 2, 0).reshape(cav_num, C, H, W)


        # # # --- 测量 channelfuse 时间 的版本---
        # import time
        # print('车辆数', cav_num)
        # torch.cuda.synchronize()
        # start_fuse = time.time()
        
        # fused = self.channelfuse(fused)
        
        # torch.cuda.synchronize()
        # end_fuse = time.time()
        # print(f"[Time] channelfuse: {(end_fuse - start_fuse) * 1000:.4f} ms")
        # # out =fused
        
        # # scaledatt
        # fused_reshaped = fused.view(cav_num, C, H * W).permute(2, 0, 1)
        
        # # --- 测量 attention 时间 ---
        # torch.cuda.synchronize()
        # start_att = time.time()
        
        # out = self.attention(fused_reshaped)
        
        # torch.cuda.synchronize()
        # end_att = time.time()
        # print(f"[Time] attention: {(end_att - start_att) * 1000:.4f} ms")
        
        # out = out.permute(1, 2, 0).reshape(cav_num, C, H, W)
        # # -----------


        # attentionwindow
        # out = self.attentionwindow(fused)

        return out

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class Select_AttFusion(nn.Module):
    def __init__(self, feature_dims,  H=None, W=None, chunk_size=None):
        super(Select_AttFusion, self).__init__()

        self.att = ScaledDotProductAttention(feature_dims)

        # mamba
        # self.att = VMAMBA2Block(feature_dims,input_resolution=[H,W],ssd_chunk_size=chunk_size)

        #linear
        #self.att = LinearAttention(feature_dims)

        self.channel_selection = ChannelSelection(feature_dims=feature_dims,  H=H, W=W)

    def forward(self, xx, record_len, confidence_maps=None):
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
            x = self.channel_selection(tensor_list, confidence_maps)

            # 更新 cav_num，因为 channel_selection 可能丢弃了车辆 (drop neighbor vehicles)
            cav_num = x.shape[0]

            # att
            # print('x', x.shape)
            x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)  # C, W, H before

            # # mamba
            # x = x.view(1, C, -1).permute(0,2,1)
            # #print('x before att',x.shape)
            # h = self.att(x, H, W)
            # h = h.permute(0, 2, 1).view(1, C, H, W)
            # #print('x after att',x.shape)


            out.append(h[0, ...])

        out = torch.stack(out)
        return out




# =================================================================================================================================================================





import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time
from typing import List, Dict, Any





# 就目前而言C_out实际为self.C_out = num_vehicles* self.C_orig * self.proportion。空间上是“均匀”的
# 用于确定每个通道保存的个数，所有patch为固定值。我认为后续可以继续完善，为了**Patches上集中“火力”**
class CollaborativeFusionModule(nn.Module):
    """
    一个完整的、用于多车协同感知的特征融合模块。

    本模块实现了以下高级混合策略：
    1.  **区域化管理 (Patch-based)**: 将特征图划分为多个区域进行独立处理，兼顾效率与灵活性。
    2.  **双重分数融合**: 在每个区域内，综合考虑“梯度信息”和“激活值幅度”来评估通道重要性。
    3.  **协同预算分配**: 作为决策中心，使用Softmax策略为每辆车在每个区域动态分配“贡献名额”（即应传输的通道数）。
    4.  **按需可变贡献**: 每辆车根据分配到的名额，只打包并传输对应数量的、最重要的通道信息。
    5.  **智能合并重建**: 在接收端，通过“基于全局分数的贪婪填充”策略，智能地处理来自多车的通道数据，
        解决信道冲突，并从所有候选通道中选出最优组合，重建出高质量的融合特征图。
    """
    def __init__(self, C_orig: int, C_out: int, H: int, W: int, patch_size: int, score_fusion_weights: List[float] = [0.5, 0.5], proportion=0.5):
        """
        初始化模块参数

        Args:
            C_orig (int): 输入的原始特征通道数。
            C_out (int): 最终输出的特征通道数。
            H (int): 特征图高度。
            W (int): 特征图宽度。
            patch_size (int): 每个正方形区域的边长。
            score_fusion_weights (List[float]): 融合梯度分数和激活值分数的权重 [w_grad, w_act]。
        """
        super().__init__()
        
        print("--- 初始化 CollaborativeFusionModule ---")

        
        self.proportion=proportion

        # 1. 保存维度参数
        self.C_orig = C_orig
        self.C_out = C_out
        self.H = H
        self.W = W
        self.P = patch_size
        
        print('H W',H,W, patch_size)
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError("特征图尺寸H和W必须能被 patch_size 整除")
        
        self.num_patches_h = H // patch_size
        self.num_patches_w = W // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # 2. 定义用于计算梯度的卷积核
        # 水平梯度核
        gx_kernel = torch.tensor([[-1., 0., 1.]], dtype=torch.float32).view(1, 1, 1, 3)
        # 垂直梯度核
        gy_kernel = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float32).view(1, 1, 3, 1)
        
        # 注册为buffer，它们是模型状态的一部分但不是可训练参数
        self.register_buffer('gx_kernel', gx_kernel)
        self.register_buffer('gy_kernel', gy_kernel)
        
        # 3. 保存分数融合权重
        self.w_grad, self.w_act = score_fusion_weights
        print(f"参数: C_orig={C_orig}, C_out={C_out}, H={H}, W={W}, PatchSize={patch_size}")
        print(f"区域数量: {self.num_patches} ({self.num_patches_h}x{self.num_patches_w})")
        print(f"分数融合权重: w_grad={self.w_grad}, w_act={self.w_act}")
        print("--- 初始化完毕 ---\n")

        # 你可以根据需要自行设计或训练这个核
        smooth_kernel_size = 3
        smooth_kernel = torch.ones(1, 1, smooth_kernel_size, smooth_kernel_size) / (smooth_kernel_size ** 2)
        self.smooth_kernel = nn.Parameter(smooth_kernel, requires_grad=False)

    def _compute_local_artifacts(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        【修改版】步骤一：本地计算。
        只计算分数，不再对特征本身进行排序。
        """
        # v1
        with torch.no_grad():

            fm_reshaped = feature_map.unsqueeze(1)

            s2_scores = torch.abs(feature_map)
            # s2_scores = F.conv2d(fm_reshaped, self.smooth_kernel, padding='same').squeeze(1)

            # grad_x = F.conv2d(fm_reshaped, self.gx_kernel, padding='same')
            # grad_y = F.conv2d(fm_reshaped, self.gy_kernel, padding='same')

            # s1_scores = (torch.abs(grad_x) + torch.abs(grad_y)).squeeze(1)
            # s1_scores = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6).squeeze(1)


            # # # 使用 PyTorch 的 min() 和 max() 函数
            # s1_scores = (s1_scores - s1_scores.min()) / (s1_scores.max() - s1_scores.min())
            # s2_scores = (s2_scores - s2_scores.min()) / (s2_scores.max() - s2_scores.min())
            # s1_scores = self.w_grad * s1_scores + self.w_act * s2_scores


            # # Max
            # max_across_channels = torch.max(fm_reshaped, dim=1, keepdim=True)[0]
            # C = fm_reshaped.shape[1]
            # s1_scores = max_across_channels.expand(-1, C, -1, -1).squeeze(1)
            

            # # 局部标准差
            # window_size = 3
            # padding = window_size // 2
            # # 局部均值：对每个通道独立进行平均池化
            # local_mean = F.avg_pool2d(fm_reshaped, kernel_size=window_size, stride=1, padding=padding)
            # # 局部方差： E[(X - E[X])^2] = E[X^2] - (E[X])^2
            # # 计算 X^2 的局部均值
            # feature_map_squared = fm_reshaped.pow(2)
            # local_mean_squared = F.avg_pool2d(feature_map_squared, kernel_size=window_size, stride=1, padding=padding)
            # # 计算局部方差
            # local_variance = local_mean_squared - local_mean.pow(2)
            # # 确保方差非负，防止浮点数误差导致微小的负值
            # local_variance = torch.clamp(local_variance, min=0)
            # # 局部标准差，形状为 (1, C, H, W)
            # local_std = torch.sqrt(local_variance)
            # # 移除批次维度，得到 (C, H, W)
            # s1_scores = local_std.squeeze(1)

            # avg
            # 在 dim=1 (通道维度) 上取平均值，得到 (1, 1, H, W)
            mean_across_channels = torch.mean(fm_reshaped, dim=1, keepdim=True)
            # 将 (1, 1, H, W) 广播到 (1, C, H, W)，然后移除批次维度得到 (C, H, W)
            C = fm_reshaped.shape[1]
            s1_scores = mean_across_channels.expand(-1, C, -1, -1).squeeze(1)



            s3_scores =  self.w_act * s2_scores  #self.w_grad * s1_scores +
            # # print(s3_scores.shape)



            
            s3_patched = rearrange(s3_scores, 'c (nh p1) (nw p2) -> (nh nw) c p1 p2', p1=self.P, p2=self.P)
            
            # 元数据计算保持不变   **补实验， mean/max/topk(先选topk>阈值的树，再平均)? 等等 谁能最准确反应patch下通道的重要性
            # # 1.
            k_for_metadata =  max(1, self.C_orig // 8 ) #1 #
            topk_scores_for_metadata, _ = torch.topk(s3_patched, k=k_for_metadata, dim=1)
            metadata_vector = torch.mean(topk_scores_for_metadata, dim=[1, 2, 3])
            # 2. 最大值
            # patch_scores_per_pixel, _ = torch.max(s3_patched, dim=1)
            # metadata_vector = torch.mean(patch_scores_per_pixel, dim=[1, 2])
            # 3. 平均
            # patch_scores_per_pixel = torch.mean(s3_patched, dim=1)
            # metadata_vector = torch.mean(patch_scores_per_pixel, dim=[1, 2])

            # 新增：计算并返回每个Patch内部的通道平均分，用于后续动态排序
            avg_channel_scores_per_patch = torch.mean(s3_patched, dim=[2, 3]) # [num_patches, C_orig]

        return {
            'metadata_vector': metadata_vector, # 特征图的元分数
            'original_feature_map': feature_map, # 返回原始的、未排序的特征图
            'avg_channel_scores_per_patch': avg_channel_scores_per_patch, # 返回用于排序的分数
        }

    # # 仅通道
    def _run_decision_logic(self, all_metadata: List[torch.Tensor], confidence_maps: torch.Tensor) -> torch.Tensor:
        # 预算溢出再分配逻辑保持不变，因为它是目前最优的方案
        stacked_metadata = torch.stack(all_metadata, dim=0)
        device = stacked_metadata.device
        C_orig_tensor = torch.tensor(self.C_orig, device=device)

        budget_proportions = torch.softmax(stacked_metadata, dim=0)
        contributions_float = budget_proportions * self.C_out
        
        excess_allocations = F.relu(contributions_float - C_orig_tensor)
        capped_contributions = torch.min(contributions_float, C_orig_tensor)
        total_excess_per_patch = torch.sum(excess_allocations, dim=0)
        
        recipient_mask = (contributions_float < C_orig_tensor).float()
        recipient_scores = stacked_metadata * recipient_mask
        sum_recipient_scores = torch.sum(recipient_scores, dim=0, keepdim=True)
        recipient_proportions = recipient_scores / (sum_recipient_scores + 1e-8)
        
        bonus_allocations = recipient_proportions * total_excess_per_patch.unsqueeze(0)
        
        final_float_contributions = capped_contributions + bonus_allocations
        
        initial_contributions = torch.floor(final_float_contributions)
        remaining_budget = self.C_out - initial_contributions.sum(dim=0)
        remainders = final_float_contributions - initial_contributions
        ranks = torch.argsort(torch.argsort(remainders, dim=0, descending=True), dim=0)
        correction_matrix = (ranks < remaining_budget).int()
        final_contribution_map = initial_contributions + correction_matrix
        
        return final_contribution_map.int()

    # # 通道&空间
    # def _run_decision_logic(self, all_metadata: List[torch.Tensor], confidence_maps: torch.Tensor) -> torch.Tensor:
    #     """

    #     Args:
    #         all_metadata (List[torch.Tensor]): 车辆元数据列表。
    #         scene_total_budget (float): 【新】整个场景期望保留的有效通道总数。
    #                                     这个值在forward中动态计算。
    #     """
    #     scene_total_budget = self.C_out * self.num_patches

    #     stacked_metadata = torch.stack(all_metadata, dim=0)
    #     device = stacked_metadata.device
    #     C_orig_tensor = torch.tensor(self.C_orig, device=device)

    #     # ==================== 阶段2.1: 空间预算动态分配 ====================
        
    #     # 1. 计算每个Patch的全局重要性分数
    #     # 1. 【新】处理外部置信度图，得到每个Patch的全局重要性分数
    #     #    confidence_maps: [N, 2, H, W]
    #     #    类别1是背景，类别0是前景
    #     foreground_confidence = confidence_maps[:, 0, :, :] # -> [N, H, W]
    #     # foreground_confidence = confidence_maps.sigmoid().max(dim=1)[0]  # 形状为 [N, H, W]
        
    #     #    将置信度图切分成Patch，并计算每个Patch的平均置信度
    #     #    'n (nh p1) (nw p2)' -> 'n (nh nw) (p1 p2)'
    #     #    [N, H, W] -> [N, num_patches, patch_area]
    #     confidence_patched = rearrange(foreground_confidence, 'n (nh p1) (nw p2) -> n (nh nw) (p1 p2)', 
    #                                    p1=self.P, p2=self.P)
        
    #     #    per_vehicle_patch_importance: [N, num_patches]
    #     #    - 每辆车对每个Patch重要性的判断（基于其分类结果）
    #     per_vehicle_patch_importance = torch.mean(confidence_patched, dim=2)

    #     #    patch_global_importance: [num_patches]
    #     #    - 将所有车辆的判断融合（求和），得到最终的全局Patch重要性
    #     patch_global_importance = torch.sum(per_vehicle_patch_importance, dim=0)

    #     # 2. 【核心修正】直接使用传入的 scene_total_budget 作为要分配的总预算。
    #     #    不再错误地乘以 self.num_patches。
    #     total_budget_to_distribute = scene_total_budget

    #     # 3. 将Patch重要性转为预算分配比例
    #     # ==================== 核心修改：新的预算比例计算方式 ====================

    #     # softmax
    #     patch_budget_proportions = torch.softmax(patch_global_importance, dim=0)
        
    #     # # 线性归一化
    #     # self.distribution_alpha = 1
    #     # # 1. 计算“完全均匀分布”的比例
    #     # #    每个Patch都获得完全相同的比例
    #     # uniform_proportions = torch.ones_like(patch_global_importance) / self.num_patches

    #     # # 2. 计算“线性归一化”的比例
    #     # total_importance = torch.sum(patch_global_importance)
    #     # #    处理总分为0的边缘情况，此时退化为均匀分布
    #     # if total_importance > 1e-8:
    #     #     linear_norm_proportions = patch_global_importance / total_importance
    #     # else:
    #     #     linear_norm_proportions = uniform_proportions
            
    #     # # 3. 【新策略】使用alpha进行混合
    #     # #    alpha*均匀 + (1-alpha)*线性
    #     # patch_budget_proportions = self.distribution_alpha * uniform_proportions + \
    #     #                            (1 - self.distribution_alpha) * linear_norm_proportions

    #     # =======================================================================

    #     # 4. 计算每个Patch应分配的浮点数预算
    #     dynamic_c_out_float = patch_budget_proportions * total_budget_to_distribute

    #     # 5. 将浮点数预算精确地转为整数预算向量，确保总和不变
    #     initial_patch_budgets = torch.floor(dynamic_c_out_float)
    #     # .item() 用于处理0维张量
    #     remaining_total_budget = total_budget_to_distribute - torch.sum(initial_patch_budgets)
    #     patch_remainders = dynamic_c_out_float - initial_patch_budgets
    #     ranks_patch = torch.argsort(torch.argsort(patch_remainders, descending=True))
    #     patch_correction = (ranks_patch < remaining_total_budget.item()).int()
    #     dynamic_c_out_int_vector = (initial_patch_budgets + patch_correction).int()
        
    #     # ==================== 阶段2.2: 车辆贡献名额分配 ====================
    #     # (此部分逻辑不变，因为它操作的基础——dynamic_c_out_int_vector——现在是正确的了)
        
    #     vehicle_proportions = torch.softmax(stacked_metadata, dim=0)
    #     contributions_float = vehicle_proportions * dynamic_c_out_int_vector.unsqueeze(0)

    #     excess_allocations = F.relu(contributions_float - C_orig_tensor)
    #     capped_contributions = torch.min(contributions_float, C_orig_tensor)
    #     total_excess_per_patch = torch.sum(excess_allocations, dim=0)
        
    #     recipient_mask = (contributions_float < C_orig_tensor).float()
    #     recipient_scores = stacked_metadata * recipient_mask
    #     sum_recipient_scores = torch.sum(recipient_scores, dim=0, keepdim=True)
    #     recipient_proportions = recipient_scores / (sum_recipient_scores + 1e-8)
        
    #     bonus_allocations = recipient_proportions * total_excess_per_patch.unsqueeze(0)
    #     final_float_contributions = capped_contributions + bonus_allocations
        
    #     initial_contributions = torch.floor(final_float_contributions)
    #     remaining_budget = dynamic_c_out_int_vector - initial_contributions.sum(dim=0)
    #     remainders = final_float_contributions - initial_contributions
        
    #     ranks_vehicle = torch.argsort(torch.argsort(remainders, dim=0, descending=True), dim=0)
    #     correction_matrix = (ranks_vehicle < remaining_budget).int()
    #     final_contribution_map = initial_contributions + correction_matrix
        
    #     return final_contribution_map.int()


    def _prepare_payload(self, artifacts: Dict[str, torch.Tensor], contribution_vector: torch.Tensor) -> torch.Tensor:
        """
        【修改版】步骤三：为单车生成稀疏特征图。
        根据实时的分数排序结果，对原始特征进行动态掩码。
        """
        original_feature_map = artifacts['original_feature_map']
        avg_channel_scores = artifacts['avg_channel_scores_per_patch'] # [num_patches, C_orig]
        
        # --- 3.1. 对原始特征分解为Patch ---
        f_patched = rearrange(original_feature_map, 'c (nh p1) (nw p2) -> (nh nw) c p1 p2', 
                              p1=self.P, p2=self.P) # [num_patches, C_orig, P, P]
                              
        # --- 3.2. 动态计算排序和掩码 (向量化实现) ---
        # 得到每个Patch内部，通道按重要性从高到低的索引
        sorted_indices = torch.argsort(avg_channel_scores, dim=1, descending=True) # [num_patches, C_orig]
        
        # 再用两次argsort得到每个原始通道的“排名”
        # ranks[i, j] 的值代表Patch i中，原始通道j的重要性排名（0最高，1次之）
        ranks = torch.argsort(sorted_indices, dim=1) # [num_patches, C_orig]
        
        # 根据分配的名额k，生成掩码。排名 < k 的通道被保留
        channel_mask = ranks < contribution_vector.unsqueeze(1) # [num_patches, C_orig]
        
        # --- 3.3. 应用掩码 ---
        mask_expanded = rearrange(channel_mask, 'np c -> np c 1 1')
        # 直接在原始顺序的Patch上应用掩码
        sparse_patches = f_patched * mask_expanded
        
        # --- 3.4. 拼合回完整大图 ---
        sparse_feature_map = rearrange(sparse_patches, '(nh nw) c p1 p2 -> c (nh p1) (nw p2)', 
                                       nh=self.num_patches_h, nw=self.num_patches_w)
        return sparse_feature_map

    # ==================== 核心修改：完全按照您的拼接思路重写 ====================
    def _reconstruct_from_payloads(self, all_payloads: List[Dict[str, torch.Tensor]], contribution_map: torch.Tensor) -> torch.Tensor:
        """
        【最终优化版】步骤四：在自车处，提取所有非空通道并直接拼接。
        此版本通过预分配内存和切片赋值，优化了循环内部的性能。
        
        输入:
            all_payloads (List[Dict]): N个车辆的数据包列表。
            contribution_map (Tensor): [N, num_patches] 的贡献表，用于确定k值。
        
        输出:
            (Tensor): 最终融合后的特征图，大小为 [1, C_out, H, W]。
        """
        num_vehicles = len(all_payloads)
        
        # 初始化一个最终的、大的Patch集合张量，避免在循环中stack
        final_map_patches = torch.zeros(self.num_patches, self.C_out, self.P, self.P, device=self.gx_kernel.device)

        # --- 对每个Patch独立进行重建 ---
        for patch_idx in range(self.num_patches):
            
            # 初始化一个指针，用于在目标Patch的通道维度上进行填充
            current_channel_idx = 0
            
            # 遍历所有车辆
            for vehicle_idx in range(num_vehicles):
                # 获取该车、该Patch被分配的贡献数k
                k = contribution_map[vehicle_idx, patch_idx].item()
                
                if k > 0:
                    # 从该车的payload中，提取出对应的整个patch
                    vehicle_patch_data = all_payloads[vehicle_idx]['masked_values'][patch_idx] # [C_orig, P, P]
                    
                    # 提取前k个非空通道
                    non_empty_channels = vehicle_patch_data[:k, :, :] # [k, P, P]
                    
                    # --- 核心优化点 ---
                    # 不再使用 list.append 和 torch.cat
                    # 而是直接将提取出的数据块，通过切片赋值放入预先分配好的final_patch中
                    start = current_channel_idx
                    end = current_channel_idx + k
                    final_map_patches[patch_idx, start:end, :, :] = non_empty_channels
                    
                    # 更新指针
                    current_channel_idx = end

        # --- 将所有重建好的Patch拼合回完整大图 ---
        final_map = rearrange(final_map_patches, '(nh nw) c p1 p2 -> 1 c (nh p1) (nw p2)', 
                            nh=self.num_patches_h, nw=self.num_patches_w)
        return final_map

    def forward(self, feature_list: List[torch.Tensor], confidence_maps=None) -> torch.Tensor:
        """
        模块主入口函数 (保持模块化风格)。
        
        输入:
            feature_list (List[Tensor]): N辆车的原始特征图列表, 每个为 [1, C_orig, H, W]。
                                         
        输出:
            (Tensor): N个稀疏特征图的堆栈, 大小为 [N, C_orig, H, W]。
        """
 
        if self.training:
            self.proportion=random.uniform(0, 0.6)
            if random.random() < 0.2:
                self.proportion = random.random() * random.random()


        # print(f"--- 模块前向传播开始，当前比例: {self.proportion} ---")


        num_vehicles = len(feature_list)

        self.C_out = num_vehicles* self.C_orig * self.proportion # 每辆车每个patch的总和 对应的通道贡献个数
        #self.C_out = self.C_orig
        

        # --- 阶段一: 所有车辆并行进行本地计算 ---
        all_local_artifacts = [self._compute_local_artifacts(f.squeeze(0)) for f in feature_list]
        
        # --- 阶段二: 收集元数据并进行中央决策 ---
        all_metadata_vectors = [artifacts['metadata_vector'] for artifacts in all_local_artifacts]

        contribution_map = self._run_decision_logic(all_metadata_vectors, confidence_maps) # [N, num_patches]
        # contribution_map = self._run_decision_logic(all_metadata_vectors) # [N, num_patches]


        # # --- 阶段一: 所有车辆并行进行本地计算 (测速) ---
        # torch.cuda.synchronize()
        # start_t1 = time.time()
        
        # all_local_artifacts = [self._compute_local_artifacts(f.squeeze(0)) for f in feature_list]
        
        # torch.cuda.synchronize()
        # end_t1 = time.time()
        # print(f"[Stage 1] Local Compute: {(end_t1 - start_t1) * 1000:.4f} ms")
        
        # # --- 阶段二: 收集元数据并进行中央决策 (测速) ---
        # all_metadata_vectors = [artifacts['metadata_vector'] for artifacts in all_local_artifacts]
        # torch.cuda.synchronize()
        # start_t2 = time.time()
        
        # contribution_map = self._run_decision_logic(all_metadata_vectors, confidence_maps) # [N, num_patches]
        
        # torch.cuda.synchronize()
        # end_t2 = time.time()
        # print(f"[Stage 2] Central Decision: {(end_t2 - start_t2) * 1000:.4f} ms")




        # # # ==================== 您要求的打印逻辑开始 ====================
        # #
        # # 在这里，我们可以打印出任何一个我们感兴趣的Patch的配额详情。
        # #
        # # 选择要检查的patch的索引，例如第一个patch (索引为0)
        # patch_to_inspect_idx = 10

        # print(f"\n--- Patch {patch_to_inspect_idx} 的通道配额详情 ---")
        
        # # 从 contribution_map 中提取该patch下所有车辆的贡献数
        # # contribution_map 的形状是 [N, num_patches]，所以我们取所有行(:)的第 patch_to_inspect_idx 列
        # contributions_for_patch = contribution_map[:, patch_to_inspect_idx]
        
        # total_channels_for_patch = 0
        # # 遍历所有车辆
        # for i in range(num_vehicles):
        #     # .item() 用于从单个元素的张量中提取出Python数值，方便打印
        #     vehicle_contribution = contributions_for_patch[i].item()
        #     print(f"车辆 {i} 的贡献数: {vehicle_contribution}")
        #     total_channels_for_patch += vehicle_contribution
            
        # print("------------------------------------")
        # print(f"所有车辆在该Patch的总贡献数: {total_channels_for_patch}")
        # # 我们可以与 self.C_out 比较，来验证预算分配是否精确
        # # 注意 self.C_out 可能是浮点数，所以取整后比较
        # print(f"该Patch的理论总预算 (self.C_out): {int(self.C_out)}")
        # print("====================================\n")
        # # # ==================== 打印逻辑结束 ====================





        # --- 阶段三: 各车根据决策生成自己的稀疏特征图 ---
        sparse_feature_maps = []

        # 引入车辆丢弃概率 (drop_prob)
        # 仅在训练且非Ego车辆时生效，模拟通信丢包或未连接
        drop_prob = 0.0
        if self.training:
            # 这里定义丢弃概率，例如 0.2 表示 20% 的概率丢弃某个邻居车
            drop_prob = 0.2 
        
        for i in range(num_vehicles):
            artifacts = all_local_artifacts[i]

            # corrected index from 10 to 0 for ego
            is_ego = (i == 0)

            # 丢弃逻辑 Check if we should drop this vehicle
            # 只有邻居车会被丢弃
            if not is_ego and random.random() < drop_prob:
                # 真正丢弃该车，不加入列表，模拟彻底断连
                continue

            if is_ego:
                # 直接使用其原始的、完整的特征图
                # 我们在本地计算时已经保存了它
                full_map = artifacts['original_feature_map']
                sparse_feature_maps.append(full_map)
            
            # 如果是其他协作车辆
            else:
                # 按照原计划，为它们生成稀疏特征图
                contribution_vector = contribution_map[i, :]
                sparse_map = self._prepare_payload(artifacts, contribution_vector)
                sparse_feature_maps.append(sparse_map)
        


        # --- 阶段四: 将所有稀疏特征图堆叠成最终输出 ---
        # 输出: [N, C_orig, H, W]
        return torch.stack(sparse_feature_maps, dim=0)


# # --- 主仿真流程 ---
# if __name__ == '__main__':
#     # --- 1. 定义超参数 ---
#     NUM_VEHICLES = 3
#     C_ORIG = 256
#     C_OUT = 256
#     H, W = 176,50
#     PATCH_SIZE = 2
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     # --- 2. 实例化我们的融合模块 ---
#     fusion_module = CollaborativeFusionModule(
#         C_orig=C_ORIG,
#         C_out=C_OUT,
#         H=H,
#         W=W,
#         patch_size=PATCH_SIZE
#     ).to(DEVICE)
#     fusion_module.eval() # 设置为评估模式

#     # --- 3. 准备输入数据 ---
#     # 模拟N辆车传来各自的BEV特征图
#     input_features = [
#         torch.randn(1, C_ORIG, H, W).to(DEVICE) for _ in range(NUM_VEHICLES)
#     ]
#     print(f"输入数据准备完毕: {len(input_features)} 辆车, 每辆车特征形状: {input_features[0].shape}")

#     #final_map = fusion_module(input_features)
#     # --- 4. 执行前向传播 ---
#     start_time = time.time()
#     final_map = fusion_module(input_features)
#     duration = time.time() - start_time
    
#     # --- 5. 打印结果 ---
#     print("="*50)
#     print("最终结果")
#     print("="*50)
#     print(f"融合模块执行总耗时: {duration:.4f}s")
#     print(f"最终输出的特征图形状: {final_map.shape}")

#     # 验证输出形状是否正确
#     assert final_map.shape == (1, C_OUT, H, W)
#     print("输出形状正确！仿真成功！")


# =================================================================================================================================================================



class SEBlock(nn.Module):
    """标准的Squeeze-and-Excitation模块。"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 此函数现在只返回权重，不与x相乘
        return self.fc(self.avg_pool(x))




class MultiScaleExpert(nn.Module):
    """
    专家网络的修正版 (V2)，完全遵循您的最新设计。
    每个并行分支都输出完整的out_channels，最后通过1x1卷积进行融合。
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化多尺度专家模块。
        :param in_channels: int, 输入通道数 (即 C/G)。
        :param out_channels: int, 期望的最终输出通道数 (通常也设为 C/G)。
        """
        super(MultiScaleExpert, self).__init__()

        # --- 三个并行的多尺度特征提取分支 ---
        # 每个分支都输出完整的 out_channels
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # --- 最终的融合层 ---
        # 1x1卷积，用于融合拼接后的特征，并将通道数从 3*out_channels 降维回 out_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状 (N, in_channels, H, W)。
        :return: 输出张量，形状 (N, out_channels, H, W)。
        """
        # -> 输入 x: (N, in_channels, H, W)

        # 1. 并行计算三个分支的输出
        out3x3 = self.branch3x3(x)
        # <- 输出 out3x3: (N, out_channels, H, W)
        
        out5x5 = self.branch5x5(x)
        # <- 输出 out5x5: (N, out_channels, H, W)
        
        out7x7 = self.branch7x7(x)
        # <- 输出 out7x7: (N, out_channels, H, W)

        # 2. 沿通道维度拼接结果
        combined_features = torch.cat([out3x3, out5x5, out7x7], dim=1)
        # <- 输出 combined_features: (N, out_channels * 3, H, W)
        
        # 3. 通过1x1卷积进行特征融合和降维
        final_output = self.fusion_conv(combined_features)
        # <- 最终输出 final_output: (N, out_channels, H, W)
        
        return final_output+x



class ChannelFusion(nn.Module):
    """
    最终版本：实现了您的全部设计。
    包括动态硬性分组、物理排序、专家处理、顺序还原，
    以及并行的SE全局权重调节。
    """
    def __init__(self, channels, groups=4, reduction=8):
        super(ChannelFusion, self).__init__()
        
        if channels % groups != 0:
            raise ValueError("通道数 'channels' 必须能被 'groups' 整除。")
            
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        
        # --- 路径A：动态分组处理 ---
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * groups, kernel_size=1)
        )
        self.experts = nn.ModuleList([
            #nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, padding=1)
            #ScaledDotProductAttention(self.group_channels)
            MultiScaleExpert(self.group_channels, self.group_channels)
            for _ in range(self.groups)
        ])
        
        # --- 路径B：全局通道上下文 ---
        self.se_block = SEBlock(channels)  
        

    def forward(self, x, temperature=1.0, is_training=True):
        """
        前向传播。
        注释约定:
        N: batch_size
        C: channels
        H: height
        W: width
        G: groups
        """
        # -> 输入 x: (N, C, H, W)
        cav_nums = x.shape[0]
        
        # ==================== 并行路径计算 ====================
        
        # --- 路径B: 在最开始时生成全局通道调节参数 ---
        se_weights = self.se_block(x)
        # <- 输出 se_weights: (N, C, 1, 1)


        
        # --- 路径A: 动态分组处理流程 ---
        # # 1. 路由器生成分配分数
        # routing_logits = self.router(x)
        # # <- 输出 routing_logits: (N, C * G, 1, 1)
        
        # routing_logits = routing_logits.view(cav_nums, self.channels, self.groups)
        # # <- 重塑后 routing_logits: (N, C, G)

        # # 2. 可微分硬性指派
        # if is_training:
        #     routing_matrix = F.gumbel_softmax(routing_logits, tau=temperature, hard=True, dim=-1)
        # else:
        #     route_indices = torch.argmax(routing_logits, dim=-1)
        #     routing_matrix = F.one_hot(route_indices, num_classes=self.groups).float()
        # # <- 输出 routing_matrix: (N, C, G)，内容是one-hot向量

        # # 3. 物理排序与分组
        # group_ids = torch.argmax(routing_matrix, dim=-1)
        # # <- 输出 group_ids: (N, C)，每个通道所属的组ID
        
        # permute_indices = torch.argsort(group_ids, dim=1)
        # # <- 输出 permute_indices: (N, C)，用于排序的索引
        
        # # >>> 关键步骤：提前计算用于恢复顺序的“逆排序”索引 <<<
        # unpermute_indices = torch.argsort(permute_indices, dim=1)
        # # <- 输出 unpermute_indices: (N, C)，用于恢复顺序的索引

        # # 为了使用gather，需要将permute_indices扩展到与x相同的维度
        # expanded_indices = permute_indices.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        # # <- 输出 expanded_indices: (N, C, H, W)
        
        # # 使用gather进行物理上的通道重排
        # x_permuted = torch.gather(x, 1, expanded_indices)
        # # <- 输出 x_permuted: (N, C, H, W)，但通道顺序已打乱
        
        # # 现在，通道已经按组排好序，可以安全地使用chunk进行切分
        # chunks_permuted = torch.chunk(x_permuted, self.groups, dim=1)
        # # <- 输出 chunks_permuted: G个张量的元组, 每个形状为 (N, C/G, H, W)

        # 普通通道划分*
        chunks_permuted = torch.chunk(x, self.groups, dim=1)
        
        # 4. 专家处理
        processed_chunks = []
        for i in range(self.groups):
            # 对每个分组的张量应用对应的专家网络

            # reshaped = chunks_permuted[i].view(cav_nums, self.group_channels, -1).permute(2, 0, 1)

            processed_chunk = self.experts[i](chunks_permuted[i])
            # score = torch.bmm(reshaped, reshaped.transpose(1, 2)) / np.sqrt(self.group_channels)
            # attn = F.softmax(score, -1)
            # processed_chunk = torch.bmm(attn, reshaped)
            # processed_chunk = processed_chunk.permute(1, 2, 0).view(cav_nums, self.group_channels, x.shape[2], x.shape[3])

            processed_chunks.append(processed_chunk)
        # <- 输出 processed_chunks: G个张量的列表, 每个形状为 (N, C/G, H, W)
        # 5. 拼接
        x_processed_permuted = torch.cat(processed_chunks, dim=1)
        # <- 输出 x_processed_permuted: (N, C, H, W)

        x_processed_restored = x_processed_permuted

        # # >>> 关键步骤：恢复为原来的通道顺序 <<<
        # # 使用之前计算好的“逆排序”索引，将通道顺序恢复如初
        # x_processed_restored = torch.gather(
        #     x_processed_permuted, 1, 
        #     unpermute_indices.unsqueeze(-1).unsqueeze(-1).expand_as(x_processed_permuted)
        # )
        # # <- 输出 x_processed_restored: (N, C, H, W)，通道顺序已恢复
        
        # # ==================== 最终融合 ====================
        
        # # 将经过分组操作并恢复顺序的特征，与最开始生成的全局权重相乘
        final_output = x_processed_restored * se_weights
        # # <- 最终输出 final_output: (N, C, H, W)

        # reshaped = x.view(cav_nums, self.channels, -1).permute(2, 0, 1)
        # score = torch.bmm(reshaped, reshaped.transpose(1, 2)) / np.sqrt(self.channels)
        # attn = F.softmax(score, -1)
        # context = torch.bmm(attn, reshaped)
        # res = context.permute(1, 2, 0).view(cav_nums, self.channels, x.shape[2], x.shape[3])

        return final_output #+ res


# ================== 主程序：演示如何使用和学习 ==================
if __name__ == '__main__':
    # 定义超参数
    batch_size = 8
    channels = 64
    height = 32
    width = 32
    num_groups = 4

    print("--- 模块类型: FinalRouterModule (可微分硬性排序分组) ---")
    
    # 实例化模块
    hard_routing_module = ChannelFusion(channels, num_groups)
    
    # 创建输入和目标
    input_features = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    
    # 定义优化器
    optimizer = torch.optim.Adam(hard_routing_module.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\n--- 演示端到端学习 ---")
    print("执行一次训练步骤...")
    
    # 训练模式
    hard_routing_module.train()
    optimizer.zero_grad()
    
    # 前向传播 (is_training=True)
    output_features = hard_routing_module(input_features, temperature=1.0, is_training=True)
    loss = criterion(output_features, target)
    loss.backward()
    optimizer.step()
    
    print(f"训练损失值: {loss.item():.4f}")
        
    print("\n--- 演示推理步骤 ---")
    # 推理模式
    hard_routing_module.eval()
    with torch.no_grad():
        output_features_eval = hard_routing_module(input_features, is_training=False)

    print(f"输入张量形状: {input_features.shape}")
    print(f"输出张量形状: {output_features_eval.shape}")
    
    if input_features.shape == output_features_eval.shape:
        print("✅  成功！输出形状与输入形状一致。")
    else:
        print("❌  失败！输出形状与输入形状不一致。")


