import torch.nn as nn

from opencood.models.diffuser.diffuser import Diffuser
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
import torch
import numpy as np
import spconv.pytorch as spconv
import torch.nn.functional as F
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.fuse_modules.commucation import Communication
from opencood.models.sub_modules.pcr import PCR




import os
import matplotlib.pyplot as plt
def show_heatmaps(matrices, path=None, figsize=(5, 5),
                  cmap='Blues'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
                pcm = ax.imshow(matrix, cmap=cmap)

    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    plt.savefig(path,dpi=1300)


class PointPillarDiffStu(nn.Module):
    def __init__(self, args):
        super(PointPillarDiffStu, self).__init__()

        self.is_train = args['is_train']
        self.max_cav = args['max_cav']
        # Pillar VFE
        ##########################################student#########################################
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.multi_scale = args['fusion']['multi_scale']
        if self.multi_scale:
            layer_nums = args['base_bev_backbone']['layer_nums']
            num_filters = args['base_bev_backbone']['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttFusion(args['fusion']['in_channels'])


        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        self.pillar_vfe_teacher = PillarVFE(args['pillar_vfe'],
                                            num_point_features=5,
                                            voxel_size=args['voxel_size'],
                                            point_cloud_range=args['lidar_range'])
        self.scatter_teacher = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone_teacher = ResNetBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv_teacher = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor_teacher = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.multi_scale = args['fusion']['multi_scale']
        if self.multi_scale:
            layer_nums = args['base_bev_backbone']['layer_nums']
            num_filters = args['base_bev_backbone']['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules_teacher = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttFusion(num_filters[idx])
                self.fuse_modules_teacher.append(fuse_network)
        else:
            self.fuse_modules_teacher = AttFusion(args['fusion']['in_channels'])

        self.cls_head_teacher = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head_teacher = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        self.object_mask = Communication(0.501)

        self.pcr = PCR(256)

        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']

        self.diffuser_teacher = Diffuser(
            in_channels=256,
            num_blocks=[3, 5, 5],
            mid_planes=[256, 256, 256],
            strides=[1, 2, 2],
            upsample_strides=[1, 2, 4],
            num_upsample_filters=[256, 256, 256],
            attention=True
        )
        self.diffuser = Diffuser(
            in_channels=256,
            num_blocks=[3, 5, 5],
            mid_planes=[256, 256, 256],
            strides=[1, 2, 2],
            upsample_strides=[1, 2, 4],
            num_upsample_filters=[256, 256, 256],
            attention=True
        )

        self.backbone_fix_ours()
        self.weights = [0.2,1,1]

        self.current_epoch = 0

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        
        # Define student modules explicitly
        student_modules = [
            self.pillar_vfe, self.scatter, self.backbone, 
            self.fuse_modules, self.cls_head, self.reg_head
        ]
        if self.shrink_flag:
            student_modules.append(self.shrink_conv)
        if self.compression:
            student_modules.append(self.naive_compressor)

        print(f"-------- Update Epoch: {epoch} --------")
        if epoch < 15:
            # Phase 1: Train Student Backbone, Freeze Diffuser (Teacher is always frozen)
            for module in student_modules:
                for p in module.parameters():
                    p.requires_grad = True
            
            # Freeze diffuser as it is not used/trained yet
            for p in self.diffuser.parameters():
                p.requires_grad = False
                
            print(f"Mode: Training Student Backbone. Diffuser is SHIELDED (Frozen).")
            
        else:
            # Phase 2: Freeze Student Backbone, Train Diffuser
            for module in student_modules:
                for p in module.parameters():
                    p.requires_grad = False
            
            # Unfreeze diffuser
            for p in self.diffuser.parameters():
                p.requires_grad = True
                
            print(f"Mode: Freezing Student Backbone. Training Diffuser ONLY.")

        #self.backbone_fix()


    def mask_offset_loss(self, gen_offset, gen_mask, gt, grid):

        # grid
        gt_mask = gt.sum(1) != 0
        count_pos = gt_mask.sum()
        count_neg = (~gt_mask).sum()
        beta = count_neg/count_pos
        loss = F.binary_cross_entropy_with_logits(gen_mask[:,0],gt_mask.float(),pos_weight= beta)

        grid = grid * gt_mask[:,None]
        gt = gt[:,:3] - grid
        gt_ind = gt != 0
        com_loss = F.l1_loss(gen_offset[gt_ind],gt[gt_ind])
        return loss, com_loss

    def backbone_fix_ours(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe_teacher.parameters():
            p.requires_grad = False

        for p in self.backbone_teacher.parameters():
            p.requires_grad = False

        for p in self.fuse_modules_teacher.parameters():
            p.requires_grad = False

        for p in self.shrink_conv_teacher.parameters():
            p.requires_grad = False

        for p in self.cls_head_teacher.parameters():
            p.requires_grad = False

        for p in self.reg_head_teacher.parameters():
            p.requires_grad = False

        for p in self.diffuser_teacher.parameters():
            p.requires_grad = False

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.fuse_modules.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False


    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)

        spatial_features = batch_dict['spatial_features']


        voxel_features_paint = data_dict['processed_lidar_paint']['voxel_features']
        voxel_coords_paint = data_dict['processed_lidar_paint']['voxel_coords']
        voxel_num_points_paint = data_dict['processed_lidar_paint']['voxel_num_points']

        batch_dict_teacher = {'voxel_features': voxel_features_paint,
                                  'voxel_coords': voxel_coords_paint,
                                  'voxel_num_points': voxel_num_points_paint,
                                  'record_len': record_len}

        batch_dict_teacher = self.pillar_vfe_teacher(batch_dict_teacher)
        # n, c -> N, C, H, W
        batch_dict_teacher = self.scatter_teacher(batch_dict_teacher)

        double_supervise_feature = batch_dict_teacher['spatial_features']

        double_record_len = torch.tensor([i * 2 for i in record_len])
        split_x = self.regroup(double_supervise_feature, double_record_len)
        object_feature_list = []
        supervise_feature_list = []
        for i in range(len(record_len)):
            supervise_feature_list.append(split_x[i][:record_len[i], :, :, :])
            object_feature_list.append(split_x[i][record_len[i]:, :, :, :])
        object_feature = torch.cat(object_feature_list, dim=0)
        supervise_feature = torch.cat(supervise_feature_list, dim=0)

        batch_dict_teacher['spatial_features'] = supervise_feature
        spatial_features_teacher = batch_dict_teacher['spatial_features']

        object_mask = self.object_mask(object_feature) # B 1 H W

        # labelmask
        label_mask = data_dict['label_dict']['pos_equal_one'] # [B, H, W, 2] 2:(0,1) 
        label_mask = label_mask.max(dim=-1)[0].unsqueeze(1)  # shape: (B, H, W) 2, 96, 352

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        if self.compression:
            spatial_features_teacher = self.naive_compressor_teacher(spatial_features_teacher)


        # multiscale fusion
        feature_list_teacher = self.backbone_teacher.get_multiscale_feature(spatial_features_teacher)
        feature_list = self.backbone.get_multiscale_feature(spatial_features)

        fused_feature_teacher_list = []
        fused_feature_list = []
        r_loss = None
        diff_grad_loss = torch.tensor(0.0).to(object_mask.device)

        #print((object_mask == 0).sum().item(), (object_mask == 1).sum().item(), object_mask.shape)

        for i, (fuse_module_teacher, fuse_module) in enumerate(zip(self.fuse_modules_teacher,self.fuse_modules)):
            fea_teacher, fea_tea_others = fuse_module_teacher(feature_list_teacher[i], record_len)
            fea_stu, fea_stu_others = fuse_module(feature_list[i], record_len)


            if i == 2:
                if self.current_epoch >= 15:
                    ##########################################
                    # DiT
                    mask = F.interpolate(object_mask, size=(fea_teacher.shape[2], fea_teacher.shape[3]), mode='bilinear', align_corners=False)
                    split_object_mask_diff = self.regroup(mask, record_len)
                    out = []
                    for xx in split_object_mask_diff:
                        xx = torch.any(xx, dim=0).unsqueeze(0)
                        out.append(xx)
                    object_mask_tea = torch.vstack(out).float()

                    mask = F.interpolate(label_mask, size=(fea_teacher.shape[2], fea_teacher.shape[3]), mode='bilinear',
                                        align_corners=False)
                    label_mask = mask.ne(0).bool()

                    object_mask_diff = (label_mask + object_mask_tea).ne(0).float()

                    t = torch.randint(0, 1000, (1,), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).long()

                    fea_stu, r_loss, _ = self.diffuser(fea_teacher, fea_stu, fea_stu_others, object_mask_diff, t)
                    #fea_stu, r_loss, _ = self.diffuser(fea_stu, fea_stu, fea_stu_others, object_mask_diff)  # without_teacher**
                else:
                    pass

                # diff_teacher
                # fea_teacher, r_loss_teacher, _ = self.diffuser_teacher(fea_teacher, fea_teacher, fea_tea_others, object_mask_diff, t)
                # diff_grad_loss = F.mse_loss(fea_teacher, fea_stu)
                # print('diff_grad_loss: ', diff_grad_loss.item())

            fused_feature_teacher_list.append(fea_teacher)
            fused_feature_list.append(fea_stu)


        fused_feature_teacher = self.backbone_teacher.decode_multiscale_feature(fused_feature_teacher_list)
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

        # without_teacher**
        # distill_loss = 0
        B, C, H, W = fused_feature_teacher.shape
        object_mask = F.interpolate(object_mask, size=(H, W), mode='bilinear', align_corners=False)

        split_object_mask = self.regroup(object_mask, record_len)
        out = []
        for xx in split_object_mask:
            xx = torch.any(xx, dim=0).unsqueeze(0)
            out.append(xx)
        object_mask = torch.vstack(out)

        masked_fused_feature = object_mask * fused_feature

        masked_fused_feature_teacher = object_mask * fused_feature_teacher
        distill_loss = F.mse_loss(masked_fused_feature, masked_fused_feature_teacher)


        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
            fused_feature_teacher = self.shrink_conv_teacher(fused_feature_teacher)


        # fused_feature, r_loss, _ = self.diffuser(fused_feature_teacher, fused_feature)


        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm, }

        if self.training:
            output_dict.update({
                'late_loss': r_loss + distill_loss+diff_grad_loss,
                # 'diff_grad_loss': diff_grad_loss
            })


        return output_dict



# #only for inference

#     def forward(self, data_dict):
#         voxel_features = data_dict['processed_lidar']['voxel_features']
#         voxel_coords = data_dict['processed_lidar']['voxel_coords']
#         voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
#         record_len = data_dict['record_len']

#         batch_dict = {'voxel_features': voxel_features,
#                         'voxel_coords': voxel_coords,
#                         'voxel_num_points': voxel_num_points,
#                         'record_len': record_len}
#         # n, 4 -> n, c
#         batch_dict = self.pillar_vfe(batch_dict)
#         # n, c -> N, C, H, W
#         batch_dict = self.scatter(batch_dict)
#         spatial_features = batch_dict['spatial_features']

#         if self.compression:
#             spatial_features = self.naive_compressor(spatial_features)

#         # multiscale fusion
#         feature_list = self.backbone.get_multiscale_feature(spatial_features)

#         fused_feature_list = []
#         for i, fuse_module in enumerate(self.fuse_modules):
#             fea_stu, fea_stu_others = fuse_module(feature_list[i], record_len)

#             if i == 2:
#                 # 在推理时，teacher不可用，使用fea_stu作为输入和条件
#                 # 掩码不可用，创建一个全1的掩码
#                 # 推理时时间步通常从最大值开始采样，这里简化为一步去噪(t=0)
#                 t = torch.zeros((1,), device=fea_stu.device).long()
#                 mask = torch.ones_like(fea_stu[:, :1, :, :])
                
#                 # 使用学生特征进行去噪
#                 fea_stu, _, _ = self.diffuser(fea_stu, fea_stu, fea_stu_others, mask, t)

#             fused_feature_list.append(fea_stu)

#         fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

#         if self.shrink_flag:
#             fused_feature = self.shrink_conv(fused_feature)

#         psm = self.cls_head(fused_feature)
#         rm = self.reg_head(fused_feature)

#         output_dict = {'psm': psm, 'rm': rm}

#         return output_dict

