
import torch
import torch.nn as nn
# 位置编码类
# 位置编码类（正弦位置编码）
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("If normalize is True, then scale should be specified.")
        self.scale = scale if scale is not None else 2 * torch.pi

    def forward(self, x):
        h, w = x.shape[-2:]
        mask = torch.zeros((h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[-1:, :] + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
        return pos


# 加权融合类
class WeightedFusion(nn.Module):
    def __init__(self, d_model):
        super(WeightedFusion, self).__init__()
        self.fc = nn.Linear(d_model, 1)  # 简单的全连接层用于计算特征的重要性权重

    def forward(self, ego_feat, other_feat):
        # 计算每个特征的重要性权重
        combined_feat = torch.cat([ego_feat, other_feat], dim=-1)  # 拼接特征
        weight = torch.sigmoid(self.fc(combined_feat))  # 使用 sigmoid 激活，范围在 [0, 1]
        fused_feat = weight * ego_feat + (1 - weight) * other_feat  # 加权融合
        return fused_feat


# 多尺度特征提取器类
class BEVMultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(BEVMultiScaleFeatureExtractor, self).__init__()
        # First scale: 1/2 resolution
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Second scale: 1/4 resolution
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Third scale: 1/8 resolution
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        B,H,W,C = x.shape
        x = x.permute(0,3,1,2)
        scale1_feat = self.scale1(x).permute(0,2,3,1)  # Shape: (B, H/2, W/2, C)
        scale2_feat = self.scale2(x).permute(0,2,3,1)  # Shape: (B, H/4, W/4, C)
        scale3_feat = self.scale3(x).permute(0,2,3,1)  # Shape: (B, H/8, W/8, C)
        x = x.permute(0,2,3,1)
        return [x, scale1_feat, scale2_feat, scale3_feat]