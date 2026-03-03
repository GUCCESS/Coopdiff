import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

import torch
import torch.nn as nn
import math
from torch.nn import functional as F


def normalization(channels):
    """简单的层归一化，适用于通道优先格式"""
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def conv_nd(dims, *args, **kwargs):
    """创建N维卷积层"""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def zero_module(module):
    """初始化模块权重为零"""
    for p in module.parameters():
        p.data.zero_()
    return module


class ConditionGuidedCrossAttention(nn.Module):
    """
    条件引导交叉注意力模块 - 位置编码与内容编码分离实现
    """

    def __init__(
            self,
            channels,
            num_heads=8,
            use_checkpoint=False,
            norm_first=True,
            use_positional_encoding=True,
            positional_encoding_type='learned',
            max_position_embeddings=1024,
            position_channels_ratio=1.0,  # 位置编码通道比例
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm_first = norm_first
        self.use_checkpoint = use_checkpoint
        self.use_positional_encoding = use_positional_encoding
        self.position_channels_ratio = position_channels_ratio

        # 输入特征的QKV投影
        self.qkv = conv_nd(2, channels, channels * 3, 1)

        # 条件特征的KV投影
        self.condition_kv = conv_nd(2, channels, channels * 2, 1)

        # 输出投影层
        self.proj_out = zero_module(conv_nd(2, channels, channels, 1))

        # 归一化层
        self.norm = normalization(channels)
        self.condition_norm = normalization(channels)

        # 位置编码
        if use_positional_encoding:
            position_channels = int(channels * position_channels_ratio)

            if positional_encoding_type == 'learned':
                # 可学习的位置编码
                self.position_embedding = nn.Parameter(
                    torch.zeros(1, position_channels, max_position_embeddings)
                )
                nn.init.trunc_normal_(self.position_embedding, std=0.02)
            elif positional_encoding_type == 'sinusoidal':
                # 固定的正弦位置编码
                self.register_buffer('position_embedding',
                                     self._create_sinusoidal_embeddings(position_channels, max_position_embeddings))
            else:
                raise ValueError(f"Unsupported positional encoding type: {positional_encoding_type}")

            # 位置编码投影层
            self.pos_proj = conv_nd(2, position_channels, channels, 1)

    def _create_sinusoidal_embeddings(self, channels, max_position_embeddings):
        """创建正弦位置编码"""
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe = torch.zeros(1, channels, max_position_embeddings)
        pe[0, 0::2, :] = torch.sin(position * div_term).transpose(0, 1)
        pe[0, 1::2, :] = torch.cos(position * div_term).transpose(0, 1)
        return pe

    def forward(self, x, condition):
        """
        :param x: 输入特征图 (N, C, H, W)
        :param condition: 条件特征图 (N, C, H, W)
        :return: 融合后的特征图
        """
        b, c, h, w = x.shape

        n_heads = self.num_heads
        seq_len = h * w
        head_dim = c // n_heads

        # 1. 输入预处理
        if self.norm_first:
            x_norm = self.norm(x)
            condition_norm = self.condition_norm(condition)
        else:
            x_norm = x
            condition_norm = condition

        # 2. 生成QKV
        qkv = self.qkv(x_norm)  # (N, 3C, H, W)
        q, k_x, v_x = qkv.chunk(3, dim=1)  # 各为 (N, C, H, W)

        # 3. 生成条件特征的KV
        condition_kv = self.condition_kv(condition_norm)  # (N, 2C, H, W)
        k_condition, v_condition = condition_kv.chunk(2, dim=1)  # 各为 (N, C, H, W)

        # 4. 处理位置编码（关键区别：在QKV之后添加位置信息）
        if self.use_positional_encoding:
            # 获取位置编码
            pos_emb = self.position_embedding[:, :, :seq_len]  # 形状: (1, position_channels, seq_len)
            # 动态扩展批次维度
            pos_emb = pos_emb.expand(b, -1, -1)  # 形状: (b, position_channels, seq_len)
            # 投影到相同维度并重塑为特征图格式
            pos_emb = pos_emb.reshape(b, -1, h, w)
            pos_proj = self.pos_proj(pos_emb)  # (b, c, h, w)

            # 重塑为序列形式
            q = q.reshape(b, c, seq_len)
            k_x = k_x.reshape(b, c, seq_len)
            v_x = v_x.reshape(b, c, seq_len)
            k_condition = k_condition.reshape(b, c, seq_len)
            v_condition = v_condition.reshape(b, c, seq_len)
            pos_proj = pos_proj.reshape(b, c, seq_len)

            # 拼接位置编码（而非加法）
            q = torch.cat([q, pos_proj], dim=1)  # (N, 2C, seq_len)
            k_x = torch.cat([k_x, pos_proj], dim=1)
            k_condition = torch.cat([k_condition, pos_proj], dim=1)

            # 更新通道维度
            total_dim = c * 2  # 内容维度 + 位置维度
            head_dim_total = total_dim // n_heads
        else:
            # 不使用位置编码时的处理
            q = q.reshape(b, c, seq_len)
            k_x = k_x.reshape(b, c, seq_len)
            v_x = v_x.reshape(b, c, seq_len)
            k_condition = k_condition.reshape(b, c, seq_len)
            v_condition = v_condition.reshape(b, c, seq_len)

            total_dim = c
            head_dim_total = head_dim

        # 5. 合并键值对
        k = torch.cat([k_x, k_condition], dim=2)  # (N, total_dim, seq_len*2)
        v = torch.cat([v_x, v_condition], dim=2)  # (N, c, seq_len*2)

        # 6. 调整为多头格式
        q = q.view(b, n_heads, head_dim_total, seq_len).transpose(2, 3)  # (N, n_heads, seq_len, head_dim_total)
        k = k.view(b, n_heads, head_dim_total, seq_len * 2).transpose(2, 3)  # (N, n_heads, seq_len*2, head_dim_total)
        v = v.view(b, n_heads, head_dim, seq_len * 2).transpose(2, 3)  # (N, n_heads, seq_len*2, head_dim)

        # 7. 计算注意力得分
        scale = 1.0 / math.sqrt(math.sqrt(head_dim_total))
        attn = (q * scale) @ (k.transpose(-2, -1) * scale)  # (N, n_heads, seq_len, seq_len*2)
        attn = F.softmax(attn, dim=-1)

        # 8. 应用注意力权重
        out = attn @ v  # (N, n_heads, seq_len, head_dim)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)  # 恢复原始形状

        # 9. 输出投影与残差连接
        out = self.proj_out(out)
        return x + out  # 残差连接

model = ConditionGuidedCrossAttention(
    channels=128,                # 输入特征通道数
    num_heads=8,                 # 注意力头数
    use_checkpoint=False,        # 是否使用检查点（节省内存）
    norm_first=True,             # 是否先进行归一化
    use_positional_encoding=True, # 是否使用位置编码
    positional_encoding_type='learned',  # 位置编码类型：'learned' 或 'sinusoidal'
    max_position_embeddings=88*44  # 最大位置编码长度
)

# 创建模拟输入数据
batch_size = 2
height, width = 24, 88
channels = 128

x = torch.randn(batch_size, channels, height, width)  # 输入特征图
condition = torch.randn(batch_size, channels, height, width)  # 条件特征图

# 前向传播
output = model(x, condition)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")