import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def feature_pooling(x):
    """
    对N C H W格式的特征图在N维度上进行max和avg池化，并将结果融合

    参数:
        x: 输入特征图，形状为 [N, C, H, W]

    返回:
        融合后的特征图，形状为 [1, C, H, W]
    """
    # 在N维度上进行max池化
    max_pooled, _ = torch.max(x, dim=0, keepdim=True)

    # 在N维度上进行avg池化
    avg_pooled = torch.mean(x, dim=0, keepdim=True)

    # 将max池化和avg池化的结果相加并除以2
    fused_features = (max_pooled + avg_pooled) / 2

    return fused_features


class SpatialTransformer(nn.Module):
    def __init__(self, input_channels=3, in_c_others=3, output_size=(100, 352)):
        super(SpatialTransformer, self).__init__()
        self.output_size = output_size

        self.conv = nn.Conv2d(input_channels + in_c_others, input_channels, kernel_size=1)
        # 定位网络 - 提取空间变换特征
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 在初始化时计算实际的 flatten 特征数
        with torch.no_grad():
            dummy = torch.zeros((1, input_channels, output_size[0], output_size[1]))
            dummy_out = self.localization(dummy)
            num_features = dummy_out.view(1, -1).shape[1]

        # 全连接层：预测仿射变换参数
        self.fc_loc = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 初始化为恒等变换
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, reg_ego, reg_others):
        print('reg_ego', reg_ego.shape)

        # reg_ego: [1, C, H, W]
        # reg_others: [N, C, H, W]
        reg_others = feature_pooling(reg_others)  # -> [1, C, H, W]

        # 拼接输入
        x = torch.cat((reg_ego, reg_others), dim=1)
        x = self.conv(x)

        # 提取定位特征
        xs = self.localization(x)
        print('xs', xs.shape)

        # 动态展平
        xs = xs.view(xs.size(0), -1)

        # 预测变换矩阵 theta
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)  # [batch_size, 2, 3]

        # 生成采样网格
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # 使用网格采样
        x_trans = F.grid_sample(x, grid, align_corners=False)

        # 残差连接
        x_out = reg_ego + x_trans

        return x_out