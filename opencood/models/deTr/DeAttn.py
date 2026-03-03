import math
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn.init import constant_, xavier_uniform_

from opencood.models.deTr.MSDAFunc import MSDeformAttnFunction


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels （特征图的数量）
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
                            每个特征图的每个attention上需要sample的points数量
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        # if not _is_power_of_2(_d_per_head):
        #     warnings.warn(
        #         "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
        #         "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # =============================================================================
        # 每个query在每个head每个特征图(n_levels)上都需要采样n_points个偏移点，每个点的像素坐标用(x,y)表示
        # =============================================================================
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # =============================================================================
        # 每个query用于计算注意力权重的参数矩阵
        # =============================================================================
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # =============================================================================
        # value的线性变化
        # =============================================================================
        self.value_proj = nn.Linear(d_model, d_model)

        # =============================================================================
        # 输出结果的线性变化
        # =============================================================================
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # =============================================================================
        # sampling_offsets的权重初始化为0
        # =============================================================================
        constant_(self.sampling_offsets.weight.data, 0.)
        # =============================================================================
        # thetas: 尺寸为(nheads, )，假设nheads = 8，则值为：
        # tensor([0*(pi/4), 1*(pi/4), 2*(pi/4), ..., 7 * (pi/4)])
        # 好似把一个圆切成了n_heads份，用于表示一个图的nheads个方位
        # =============================================================================
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # =============================================================================
        # grid_init: 尺寸为(nheads, 2)，即每一个方位角的cos和sin值，例如：
        # tensor([[ 1.0000e+00,  0.0000e+00],
        #         [ 7.0711e-01,  7.0711e-01],
        #         [-4.3711e-08,  1.0000e+00],
        #         [-7.0711e-01,  7.0711e-01],
        #         [-1.0000e+00, -8.7423e-08],
        #         [-7.0711e-01, -7.0711e-01],
        #         [ 1.1925e-08, -1.0000e+00],
        #         [ 7.0711e-01, -7.0711e-01]])
        # =============================================================================
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # =============================================================================
        # 第一步：
        # grid_init / grid_init.abs().max(-1, keepdim=True)[0]：计算8个head的坐标偏移，尺寸为torch.Size([n_heads, 2])
        # 结果为：
        # tensor([[ 1.,  0.],
        #         [ 1.,  1.],
        #         [0.,  1.],
        #         [-1.,  1.],
        #         [-1., 0.],
        #         [-1., -1.],
        #         [0., -1.],
        #         [1., -1]])
        # 然后把这个数据广播给每个n_level的每个n_point
        # 最后grid_init尺寸为：(nheads, n_levels, n_points, 2)
        # 这意味着：在第一个head上，每个level上，每个偏移点的偏移量都是(1,0)
        #         在第二个head上，每个level上，每个偏移点的偏移量都是(1,1)，以此类推
        # =============================================================================
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                              self.n_levels,
                                                                                                              self.n_points,
                                                                                                              1)
        # =============================================================================
        # 每个参考点的初始化偏移量肯定不能一样，所以这里第i个参考点的偏移量设为：
        # (i,0), (i,i), (0,i)...(i,-i)
        # grid_init尺寸依然是：(nheads, n_levels, n_points, 2)
        # 现在意味着：在第一个head上，每个level上，第一个偏移点偏移量是(1,0), 第二个是(2,0)，第三个是(3,0), 第四个是(4,0)
        #           在第二个head上，每个level上，都一个偏移点偏移量是(1,1), 第二个是(2,2), 第三个是(3,3), 第四个是(4,4)
        # =============================================================================
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # =============================================================================
        # 初始化sampling_offsets的bias，但其不参与训练。尺寸为(nheads * n_levels * n_points * 2,)
        # =============================================================================
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # =============================================================================
        # 其余参数的初始化
        # =============================================================================
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """


        Args:
            query：原始输入数据 + 位置编码后的结果，尺寸为(B, sum(所有特征图的token数量), 256)
                   sum(所有特征图的token数量)其实就是sum(H_*W_)

            reference_points:尺寸为(B, sum(所有特征图的token数量), level_num, 2)。表示对于
                            batch中的每一条数据的每个token，它在不同特征层上的坐标表示。
                                    请一定参见get_reference_points函数相关注释

            input_flatten： 原始输入数据，尺寸为(B, sum(所有特征图的token数量), 256)

            input_spatial_shapes: tensor，其尺寸为(level_num，2)。 表示原始特征图的大小。
                                  其中2表是Hi, Wi。例如：
                                  tensor([[94, 86],
                                          [47, 43],
                                          [24, 22],
                                          [12, 11]])

            input_level_start_index: 尺寸为(level_num, )
                                     表示每个level的起始token在整排token中的序号，例如：
                                     tensor([0,  8084, 10105, 10633])

            input_padding_mask：     mask信息，(B, sum(所有特征图的token数量))

        """
        # =============================================================================
        # N：batch_size
        # len_q: query数量，在encoder attention中等于len_in
        # len_in: 所有特征图组成的token数量
        # =============================================================================
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 声明所有特征图的像素数量 = token数量
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # =============================================================================
        # self.value_proj：线性层，可理解成是Wq, 尺寸为(d_model, d_model)
        # value：v值，尺寸为(B, sum(所有特征图的token数量), 256)
        # =============================================================================
        value = self.value_proj(input_flatten)
        # =============================================================================
        # 对于V值，将padding的部分用0填充（那个token向量变成0向量）
        # =============================================================================
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # =============================================================================
        # 将value向量按head拆开
        # value：尺寸为：(B, sum(所有特征图的token数量), nheads, d_model//n_heads)
        # =============================================================================
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # =============================================================================
        # self.sampling_offsets：偏移点的权重，尺寸为(d_model, n_heads * n_levels * n_points * 2)
        #                        【对于一个token，求它在每个head的每个level的每个偏移point上
        #                        的坐标结果(x, y)】
        # 由于sampling_offsets.weight.data被初始化为0，但sampling_offsets.bias.data却被初始化
        # 为我们设定好的偏移量，所以第一次fwd时，这个sampling_offsets是我们设定好的初始化偏移量
        # self.sampling_offsets(query) = (B, sum(所有特征图的token数量), d_model) *
        #                                (d_model, n_heads * n_levels * n_points * 2)
        #                              = (B, sum(所有特征图的token数量), n_heads * n_levels * n_points * 2)
        # =============================================================================
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # =============================================================================
        # self.attention_weights: 线性层，尺寸为(d_model, n_heads * n_levels * n_points)，
        #                         初始化时weight和bias都被设为0
        # 因此attention_weights第一次做fwd时全为0
        # =============================================================================
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # =============================================================================
        # attention_weights: 表示每一个token在每一个head的每一个level上，和它的n_points个偏移向量
        #                    的attention score。
        #                    初始化时这些attention score都是相同的
        # =============================================================================
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # =============================================================================
        # reference_points：尺寸为(B, sum(所有token数量), level_num, 2)
        # N, Len_q, n_heads, n_levels, n_points, 2
        # =============================================================================
        if reference_points.shape[-1] == 2:
            # ======================================================================
            # offset_normalizer: 尺寸为(level_num, 2)，表示每个特征图的原始大小，坐标表达为(W_, H_)
            # ======================================================================
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # ======================================================================
            # 先介绍下三个元素：
            # reference_points[:, :, None, :, None, :]:
            #  尺寸为 (B, sum(token数量), 1, n_levels, 1,2)

            # sampling_offsets:
            # 尺寸为（B, sum(token数量), n_heads, n_levels, n_points, 2）

            # offset_normalizer[None,None,None,:,None,:]:
            # 尺寸为(1, 1, 1,n_levels, 1,2)

            # 再介绍下怎么操作的：
            # （1）sampling_offsets / offset_normalizer[None, None, None, :, None, :]：
            # 前者表示预测出来的偏移量（单位是像素绝对值）通过相除，把它变成像素归一化以后的维度
            #  (2) 加上reference_points：表示把该token对应的这个参考点做偏移，
            # 得到其在各个level上的n_points个偏移结果,偏移结果同样是用归一化的像素坐标来表示
            # sampling_locations：尺寸为(B, sum(tokens数量), nhead, n_levels, n_points, 2)
            # ======================================================================

            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttn.ms_deform_attn_core_pytorch(
            value, input_spatial_shapes,  sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Args:
            spatial_shapes: tensor，其尺寸为(level_num，2)。 表示原始特征图的大小。
                            其中2表是Hi, Wi。例如：
                            tensor([[94, 86],
                                   [47, 43],
                                   [24, 22],
                                   [12, 11]])

            valid_ratios:      尺寸为(B, level_num, 2)，
                               用于表示batch中的每条数据在每个特征图上，分别沿着H和W方向的
                               有效比例（有效 = 非padding部分）
                               例如特征图如下：
                               1, 1, 1, 0
                               1, 1, 1, 0
                               0, 0, 0, 0
                               则该特征图在H方向上的有效比例 = 2/3 = 0.6
                               在W方向上的有效比例 = 3/4 = 0.75

        """
        reference_points_list = []

        #  处理每个 level下的 平面参考点，使用 归一化的适应判定某一点是否是 padding (>1的就是padding)
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # =========================================================================
            # （1）torch.linspace(0.5, H_ - 0.5, H_): 可以看成按0.5像素单位把H方向切割成若干份
            # （2）torch.linspace(0.5, W_ - 0.5, W_): 可以看成按0.5像素单位把W方向切割成若干份
            #  例如设H_, W_ = 12, 16, 则（1）和（2）分别为：
            #  tensor([ 0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,
            #           7.5000,  8.5000,  9.5000, 10.5000, 11.5000])
            #  tensor([ 0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,
            #           7.5000,  8.5000,  9.5000, 10.5000, 11.5000, 12.5000, 13.5000,
            #           14.5000, 15.5000])
            #
            # 你可以想成把一张特征图横向划几条线，纵向画几条线。对于一个像素格子，我们用其质心坐标
            # 表示它，这相当于是这些线的交界点
            #
            # 这里ref_y表示每个ref点的x坐标（H方向坐标），ref_x表示每个ref点的y坐标（W方向坐标）
            # (3) ref_y: 尺寸为(H_, W_), 形式如：
            # tensor([[ 0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            #           0.5000, 0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            #           0.5000,  0.5000],
            #         [ 1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,
            #           1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,
            #           ¥           1.5000,  1.5000],
            #         ...
            #         [11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000,
            #          11.5000,
            #          11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000,
            #          11.5000]])
            #
            # (4) ref_x：尺寸为(H_, W_)，形式如：
            # tensor([[  0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,
            #            7.5000,  8.5000,  9.5000, 10.5000, 11.5000, 12.5000, 13.5000,
            #            14.5000, 15.5000],
            #         [  0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,
            #            7.5000, 8.5000,  9.5000, 10.5000, 11.5000, 12.5000, 13.5000,
            #            14.5000, 15.5000],
            #         ...(重复下去)
            # =========================================================================
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            # =========================================================================
            # 相当于每个像素格子都用其中心点的坐标来表示它
            #  ref_y.reshape(-1)[None]: 把(H_, W_)展平成(1, H_*W_)。
            #                           例如H_=12, W_=16, 则展平成(1, 192)
            # tensor([[ 0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            #           0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            #           0.5000,  0.5000,
            #           1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,
            #           1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,  1.5000,
            #           1.5000,  1.5000,
            #           ...
            #           11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000,
            #           11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000, 11.5000,
            #           11.5000, 11.5000]])
            #
            # ref_x.reshape(-1)[None]：把(H_, W_)展平成(1, H_*W_)。例子同上
            # tensor([[ 0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,
            #           7.5000,  8.5000,  9.5000, 10.5000, 11.5000, 12.5000, 13.5000,
            #           14.5000, 15.5000, 0.5000,  1.5000,  2.5000,  3.5000,  4.5000,
            #           5.5000,  6.5000,  7.5000,  8.5000,  9.5000, 10.5000, 11.5000,
            #           12.5000, 13.5000, 14.5000, 15.5000,
            #           ...
            #
            # valid_ratios[:, None, lvl, 1]：
            # 取出batch中所有数据在当前lvl这层特征图上，H方向的有效比例（有效=非padding）
            # 尺寸为(B, 1)，例如：
            # tensor([[0.6667],
            # [0.6667],
            # [0.9167],
            # [1.0000]])
            # 乘上H_后表示实际有效的像素级长度
            # valid_ratios[:, None, lvl, 0]：也是同理
            #
            # ref_y: 尺寸为(B, H_ * W_)。
            #        表示对于batch中的每条数据，它在该lvl层特征图上一共有H_*W_个参考点，ref_y
            #        表示这些参考点最终在H方向上的像素坐标。【但这里像素坐标做了类似归一划的处理。
            #        ref_y = 原始H方向的绝对像素坐标/H方向上有效即非padding部分的绝对像素长度
            #        因此该值如果 > 1则说明该参考点在padding部分】  就是归一化
            # ref_x：尺寸为(B, H_ * W_)。同上
            # =========================================================================
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)

            # =========================================================================
            # ref：尺寸为(B, H_*W_, 2)，表示对于batch中的每条数据，它在该lvl层特征图上所有H_*W_个参考点的x,y坐标
            #     如上所说，该坐标已经处理成相对于有效像素长度的形式
            #    【特别注意！这里W和H换了位置！！！！！！】
            # =========================================================================
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # =========================================================================
        # 尺寸为：(B, sum(H_*W_), 2)。表示对于一个batch中的每一条数据，
        #       它在所有特征图上的参考点(数量为sum(H_*w))的像素坐标x,y。
        #       这里x，y都做过处理，表示该参考点相对于非padding部分的H，W的比值
        #       例如，如果x和y>1, 则说明该参考点在padding部分
        # =========================================================================
        reference_points = torch.cat(reference_points_list, 1)
        # =========================================================================
        # 尺寸为：(B, sum(H_*W_), level_num, 2)。表示对于batch中的每一条数据，
        #        它的每一个特征层上的归一化坐标，在其他特征层（也包括自己）上的所有归一化坐标
        # 假设对于某条数据：
        # 特征图1的高度为H1，有效高度为HE1，其上某个ref点x坐标为h1。则该ref点归一化坐标可以表示成h1/H1
        # 特征图2的高度为H2，有效高度为HE2，那么特征图1中的ref点在特征图2中，对应的归一化坐标应该是多少？
        # 【常规思路】：正常情况下，你可能觉得，我只要对每一张特征图上的像素点坐标都做归一化，然后对任意两张特征图，
        #             我取出像素点坐标一致的ref点，它不就能表示两张特征图的相同位置吗？
        # 【问题】：每张特征图padding的比例不一样，因此不能这么做。举例（参见草稿纸上的图）。特征图1上绝对像素
        #         位置3.5的点，在特征图2上的位置是2.1，在有效图部分之外
        # 【正确做法】：把特征图上的坐标表示成相对于有效部分的比例
        # 【解】：我们希望 h1/HE1 = h2/HE2，在此基础上我们再来求h2/H2
        #  则我们有：h2 = (h1/HE1) * HE2，进一步有
        #  (h2/H2) = (h1/HE1) * (HE2/H2)，而(h1/HE1)就是reference_points[:, :, None]，
        #                                  (HE2/H2)就是valid_ratios[:, None]
        # 所以，这里是先将不同特征图之间的映射转为“绝对坐标/有效长度”的表达，然后再转成该绝对坐标在整体长度上的比例
        # =========================================================================
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @staticmethod
    def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
        # for debug and test only,
        # need to use cuda version instead
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_,
                                                                                                         Lq_)
        return output.transpose(1, 2).contiguous()


