

import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
from opencood.visualization import simple_vis1 as simple_vis


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--save_vis_n', type=int, default=10000,
                        help='间隔多少帧保存一次可视化结果')
    parser.add_argument('--eval_epoch', type=str, default=None,
                        help='Set the checkpoint')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt

def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=8,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    if opt.eval_epoch is not None:
        epoch_id = opt.eval_epoch
        epoch_id, model = train_utils.load_saved_model(saved_path, model, epoch_id)
    else:
        epoch_id, model = train_utils.load_saved_model(saved_path, model)
    

    # pretrain_path = '/home/my/桌面/chengong/DSRC/opencood/logs/point_pillar_channelselect_50%_multi/test/net_epoch22.pth'
    # # 加载权重（严格模式关闭）
    # pre_weights = torch.load(pretrain_path, map_location=device)
    # # model.load_state_dict(pre_weights, strict=False)

    
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    iter = 0
    # Performance Analysis
    # for i, batch_data in enumerate(data_loader):
    #     batch_data = train_utils.to_device(batch_data, device)
    #     try:
    #         from thop import profile
    #         # thop 会根据 forward 的执行路径统计。
    #         # 你的 forward 函数在测试时只用了 student 模块，所以 thop 结果是准确的 student 参数量/计算量
    #         flops, params = profile(model, inputs=(batch_data['ego'],), verbose=False)
            
    #         print('\n' + '='*50)
    #         print("Performance Analysis (Student Only):")
    #         print(f"Total thop profile Estimated FLOPs: {flops / 1e9:.4f} GFLOPs")
    #         print(f"Total thop profile Active Parameters: {params / 1e6:.4f} M")
            
    #         # 验证：手动过滤掉名字里带 teacher 的参数
    #         manual_student_params = sum(p.numel() for n, p in model.named_parameters() if 'teacher' not in n)
    #         print(f"Manual Verified Student Parameters: {manual_student_params / 1e6:.4f} M")

    #         # 对比：所有加载的参数（包含 teacher）
    #         total_loaded = sum(p.numel() for p in model.parameters())
    #         print(f"\n[Ref] Total Loaded Parameters (Including Teacher): {total_loaded / 1e6:.4f} M")
            
    #         #Memory assuming float32 (4 bytes)
    #         print(f"Total Memory for ALL parameters: {total_loaded * 4 / (1024**2):.2f} MB")
    #         print('='*50 + '\n')
            
    #     except ImportError:
    #         print("Please install thop: pip install thop")
        
    #     exit()


    for i, batch_data in tqdm(enumerate(data_loader)):

        # if batch_data['ego']['record_len'].item() >= 4:

        #     iter = iter + 1
        #     print("Current CAVs are ", batch_data['ego']['record_len'].item(), " total number are",  str(iter))

        # else:

        #     continue


        # print(i)
        # if i<1500:
        #     continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)
                
            left_hand = True

            #if opt.save_vis_n and opt.save_vis_n >i:
            if i%opt.save_vis_n == 0:
                print('输出可视化结果',  i)
                
                if pred_box_tensor == None:
                    continue
                vis_save_path = os.path.join(opt.model_dir, 'vis_3d')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_3d/3d_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='3d',
                                    left_hand=left_hand,
                                    vis_pred_box=True, #False
                                    vis_gt_box=True) #False
                
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev/bev_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand,
                                    vis_pred_box=True, #False
                                    vis_gt_box=True) #False

    ap_50, ap_70 = eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)

    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        msg = 'Epoch: {} | AP @0.5: {:.04f} | AP @0.7: {:.04f}\n'.format(epoch_id,ap_50, ap_70)
        f.write(msg)
        print(msg)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
