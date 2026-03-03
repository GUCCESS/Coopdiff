import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.deTr.DeAttn import MSDeformAttn
from opencood.models.deTr.sub_modules import  WeightedFusion, BEVMultiScaleFeatureExtractor, \
    PositionEmbeddingSine


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim*2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xx = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(xx)
        return self.norm(x)



class DeCooper(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        dim, num_level, num_heads, n_poins = 256, 3, 8, 4
        H,W,dropout = 24,88,0.1
        if config is None:
            self.dim = dim
            self.num_level = num_level
            self.num_heads = num_heads
            self.n_poins = n_poins
            self.H = H
            self.W = W
            self.dropout = dropout
        else:
            self.dim = config['dim']
            self.num_level = config['num_level']
            self.num_heads = config['num_heads']
            self.n_poins = config['n_poins']
            self.H = config['H']
            self.W = config['W']
            self.dropout = config['dropout']

        self.deformable_attn = MSDeformAttn(self.dim, self.num_level, self.num_heads, self.n_poins)
        self.norm = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(self.dropout)
        self.position_embedding = PositionEmbeddingSine(self.dim // 2)

        self.mlp = MLP(self.dim)

    def forward(self, querys, feats):
        # querys 和 feats 都已经是多尺度特征
        # querys:   [[B C H W],[B C H//2 W//2],[B C H//4 W//4]]
        # feats:    [[B C H W],[B C H//2 W//2],[B C H//4 W//4]]

        feat = feats[0]

        B, C, H, W = feat.size()

        # 准备 Deformable Attention 的输入

        ms_query_location = []
        ms_feat = []
        for query, single_feat in zip(querys,feats):

            pos = self.position_embedding(single_feat)

            single_feat_loc = query + pos # B, C, H, W

            single_feat = single_feat.view(B, C, -1).permute(0,2,1)
            single_feat_loc = single_feat_loc.view(B, C, -1).permute(0,2,1)

            ms_feat.append(single_feat)
            ms_query_location.append(single_feat_loc)


        input = torch.cat(ms_feat,dim=1)  # (b, h*w, d_model)

        query_location = torch.cat(ms_query_location, dim=1)

        spatial_shapes = torch.tensor([[H,W], [H // 2, W // 2], [H // 4, W // 4]], dtype=torch.long, device=feat.device)
        # level_start_index = torch.tensor([0, (H // 2) * (W // 2), (H // 2) * (W // 2) + (H // 4) * (W // 4)],
        #                                  dtype=torch.long)

        valid_ratios = torch.ones(B, self.num_level, 2, device=feat.device)
        reference_points = MSDeformAttn.get_reference_points(spatial_shapes, valid_ratios, feat.device)
        padding_mask = None

        # （B, total, C）
        attn = self.deformable_attn(query_location, reference_points, input, spatial_shapes, None,
                                    padding_mask)

        scale0_tokens = H * W
        scale1_tokens = (H // 2) * (W // 2)
        scale2_tokens = (H // 4) * (W // 4)
        # scale3_tokens = (H // 8) * (W // 8)

        scale0_feat = attn[:, :scale0_tokens, :].view(B, H , W , C).permute(0, 3, 1,2)
        scale1_feat = attn[:, scale0_tokens:scale1_tokens+scale0_tokens, :].view(B, H // 2, W // 2, C).permute(0, 3, 1,2)  # (B, 2*C, H/2, W/2)
        scale2_feat = attn[:, scale1_tokens+scale0_tokens:scale0_tokens + scale1_tokens + scale2_tokens, :].view(B, H // 4, W // 4, C)\
                                                                                .permute(0, 3, 1,2)  # (B, 2*C, H/4, W/4)
        # scale3_feat = attn[:, scale0_tokens + scale1_tokens + scale2_tokens:, :].view(B, H // 8, W // 8, C)\
        #                                                                         .permute(0, 3, 1,2)  # (B, 2*C, H/8, W/8)

        # # 上采样每个尺度的特征图，使其与原始 ego_feat 的空间分辨率匹配
        # upsampled_scale1_feat = F.interpolate(scale1_feat, size=(H, W), mode='bilinear', align_corners=False)
        # upsampled_scale2_feat = F.interpolate(scale2_feat, size=(H, W), mode='bilinear', align_corners=False)
        # #upsampled_scale3_feat = F.interpolate(scale3_feat, size=(H, W), mode='bilinear', align_corners=False)
        #
        # fused_attn_output = (scale0_feat + upsampled_scale1_feat + upsampled_scale2_feat ) / 3  # (B, C, H, W)
        # fused_attn_output = fused_attn_output.permute(0, 2, 3, 1)
        #
        # # print(fused_attn_output.shape)
        # # print(feat.shape)
        # feat = feat + self.drop(fused_attn_output)
        #
        # out = self.mlp(self.norm(feat))

        return scale0_feat, scale1_feat, scale2_feat







