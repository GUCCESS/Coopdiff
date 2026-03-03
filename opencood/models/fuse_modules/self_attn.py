# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        # C, W, H = x.shape[1:]
        # x = x.view(1, C, -1).permute(2, 0, 1)
        # x = self.att(x, x, x)
        # h = x.permute(1, 2, 0).view(1, C, W, H)
        # return h

        split_x = self.regroup(x, record_len) #N CHW    split_x:[N1 CHW,  N2CHW]
        C, W, H = split_x[0].shape[1:]
        out = [] #自车特征
        out2 = [] #它车特征

        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)  #N1, C, W, H
            out.append(h[0, ...]) # 添加了 CHW

            if cav_num > 1:
                # 获取剩余特征 [1:]
                remaining_features = h[1:, ...]  # (cav_num-1, C, W, H)

                # 在第一个维度(特征维度)上取max和mean
                max_feature = torch.max(remaining_features, dim=0)[0]  # (C, W, H)
                mean_feature = torch.mean(remaining_features, dim=0)  # (C, W, H)
                # 将max和mean结果相加
                h2 = (max_feature + mean_feature)/2  # (C, W, H)
            else:
                # 如果只有一个特征，合并后的特征为零张量
                h2 = h[0, ...]
            out2.append(h2)

        #print(torch.stack(out).shape, torch.stack(out2).shape)
        # [CHW, CHW] -> 2CHW
        return torch.stack(out), torch.stack(out2)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
