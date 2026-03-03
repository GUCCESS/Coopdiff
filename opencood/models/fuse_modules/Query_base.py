import torch
import torch.nn as nn
import torch.nn.functional as F


# 位置编码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 注册为非训练参数

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 自注意力机制实现
def self_attention(q, k, v, mask=None):
    # 计算注意力分数
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 注意力权重
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和得到输出
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 定义线性变换层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变换并分割成多头
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 应用自注意力
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展维度以匹配多头

        attn_output, attn_weights = self_attention(q, k, v, mask)

        # 合并多头并应用最终线性变换
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_linear(attn_output)

        return output, attn_weights


# Transformer编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力和残差连接
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # 前馈网络和残差连接
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src


# Transformer解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # 交叉注意力
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 自注意力和残差连接
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        # 交叉注意力和残差连接（Query与Encoder输出交互）
        cross_attn_output, attn_weights = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)

        # 前馈网络和残差连接
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)

        return tgt, attn_weights


# 多层感知机
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 简化版DETR模型
class DETR(nn.Module):
    def __init__(self, num_classes, num_queries=100, hidden_dim=256, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()

        # 模拟CNN骨干网络（实际中应使用预训练的ResNet等）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer编码器
        encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, dim_feedforward)
            for _ in range(num_encoder_layers)
        ])
        self.encoder = nn.Sequential(*encoder_layers)

        # Transformer解码器
        decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, dim_feedforward)
            for _ in range(num_decoder_layers)
        ])
        self.decoder = nn.Sequential(*decoder_layers)

        # 可学习的目标查询向量（核心Query机制）
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 输出预测头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 表示背景类
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 输出[x, y, w, h]格式的边界框

    def forward(self, x):
        # 1. 提取图像特征
        features = self.backbone(x)  # [B, hidden_dim, H/4, W/4]

        # 2. 重塑特征以适应Transformer输入格式
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]

        # 3. 添加位置编码
        pos_encoding = self.pos_encoder(features)

        # 4. Transformer编码
        memory = self.encoder(pos_encoding)  # [B, H*W, hidden_dim]

        # 5. 准备Query向量（核心机制）
        query_embed = self.query_embed.weight  # [num_queries, hidden_dim]
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)  # [B, num_queries, hidden_dim]

        # 6. Transformer解码：Query与图像特征交互
        decoder_output = query_embed  # 初始化解码器输入为Query向量
        for decoder_layer in self.decoder:
            decoder_output, _ = decoder_layer(decoder_output, memory)

        # 7. 输出预测结果
        outputs_class = self.class_embed(decoder_output)  # [B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(decoder_output).sigmoid()  # [B, num_queries, 4]

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out


# 示例：使用模型进行推理
def demo():
    # 创建模型（假设检测80个类别）
    model = DETR(num_classes=80)

    # 模拟输入图像 [batch_size=2, channels=3, height=256, width=256]
    x = torch.randn(2, 3, 256, 256)

    # 前向传播
    outputs = model(x)

    # 输出结果：
    # outputs['pred_logits']: [2, 100, 81] - 类别预测（包含背景类）
    # outputs['pred_boxes']: [2, 100, 4] - 边界框预测 [x, y, w, h]（归一化坐标）

    print(f"预测类别形状: {outputs['pred_logits'].shape}")
    print(f"预测边界框形状: {outputs['pred_boxes'].shape}")


if __name__ == "__main__":
    demo()
