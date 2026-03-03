
import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    # output_dict['ego'] = model(cav_content)

    # pred_box_tensor, pred_score, gt_box_tensor = \
    #     dataset.post_process(batch_data,
    #                          output_dict)

    # return pred_box_tensor, pred_score, gt_box_tensor

    out_dict = model(cav_content)
    output_dict['ego'] = {
        'psm': out_dict['psm'],
        'rm': out_dict['rm'],
    }

    if 'psm_singe' in out_dict:
        cls_late_tensor = out_dict['psm_singe']  # (N, C_cls, H, W)
        reg_late_tensor = out_dict['rm_singe']  # (N, C_reg, H, W)  # 可能为 None

        # 获取 agent 数量 N
        N = cls_late_tensor.shape[0]
        print('N', N)
        for i in range(N):
            # 每个 agent 的预测是一个 (1, C, H, W) 的 tensor
            cls_i = cls_late_tensor[i].unsqueeze(0)  # (1, C_cls, H, W)
            reg_i = reg_late_tensor[i].unsqueeze(0)  # (1, C_reg, H, W)
            data_dict = {
                'psm': cls_i,
                'rm': reg_i,
            }
            
            
            output_dict[str(i)] = data_dict
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor




def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)
