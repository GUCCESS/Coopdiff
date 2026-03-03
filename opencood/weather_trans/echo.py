import argparse
import os
import time
import glob
import numpy as np
import multiprocessing as mp
import copy
from pathlib import Path
from tqdm import tqdm
# from nuscenes import NuScenes
import pickle
import random

seed = 1205
random.seed(seed)
np.random.seed(seed)


def apply_incomplete_echo_by_height(point_cloud_np: np.ndarray, drop_ratio=0.9, water_height: float = 0.001) -> np.ndarray:
    """
    根据高度筛选点云，并随机删除部分点以模拟不完整回波。

    Args:
        point_cloud_np (np.ndarray): 待处理的点云数据，形状为 (N, 4)。
        drop_ratio (float): 要删除的点的比例。
        water_height (float): 高度阈值，只有高于此值的点才会被筛选。

    Returns:
        np.ndarray: 处理后的点云数据。
    """

    np.random.seed(420)
    drop_ratio = 0.9 #0.25 0.5 0.75 0.9


    scan = point_cloud_np
    
    # 筛选出高度（z坐标）大于water_height的点
    # 假设点云数据的z坐标位于第2列（索引为2）
    height_mask = scan[:, 2] > water_height
    
    if np.sum(height_mask) > 10:
        # 获取满足条件的点的索引
        idx_to_filter = np.squeeze(np.argwhere(height_mask))
        
        # 随机选择要删除的点的数量
        num_pix_to_drop = int(len(idx_to_filter) * drop_ratio)
        idx_to_drop = np.random.choice(idx_to_filter, num_pix_to_drop, replace=False)

        # 从点云中删除这些点
        scan = np.delete(scan, idx_to_drop, axis=0)
    
    return scan


def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR Incomplete Echo')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data_root/nuScenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset',
                        default='./save_root/incomplete_echo/light')  # ['light','moderate','heavy']
    parser.add_argument('-t', '--drop_ratio', help='drop ratio of instance points', type=float, default=0.75)
    arguments = parser.parse_args()
    return arguments


# 这是用于多进程的包装函数，它负责文件I/O
def _map_processor(info: dict) -> None:
    # 从info字典中读取文件路径，这个字典是从nuscenes_infos_val.pkl加载的
    lidar_path_relative = info['lidar_path'][16:]
    lidar_path_full = os.path.join(args.root_folder, lidar_path_relative)

    # 读取点云文件
    scan = np.fromfile(lidar_path_full, dtype=np.float32, count=-1).reshape([-1, 4])  

    # 调用核心处理函数，只传入点云和参数
    modified_scan = apply_incomplete_echo_by_height(scan, args.drop_ratio)
    
    # 构建保存路径
    lidar_save_path_full = os.path.join(args.dst_folder, lidar_path_relative)
    
    # 创建父目录（如果不存在）
    os.makedirs(os.path.dirname(lidar_save_path_full), exist_ok=True)
    
    # 保存处理后的点云
    modified_scan.astype(np.float32).tofile(lidar_save_path_full)


if __name__ == '__main__':
    args = parse_arguments()
    # incomplete echo (light: 0.75, moderate: 0.85, heavy: 0.95)
    print('')
    print(f'using {args.n_cpus} CPUs')

    imageset = os.path.join(args.root_folder, "nuscenes_infos_val.pkl")
    with open(imageset, 'rb') as f:
        infos = pickle.load(f)
    all_files = infos['infos']

    # 这里我们直接将包含文件信息的字典列表传递给多进程，而不是索引
    all_paths = copy.deepcopy(all_files)
    
    # Path(args.dst_folder).mkdir(parents=True, exist_ok=True)
    
    print('Starting multi-processing...')
    n = len(all_files)
    with mp.Pool(args.n_cpus) as pool:
        # 使用 pool.imap_unordered 可以乱序处理任务，但处理速度更快
        l = list(tqdm(pool.imap(_map_processor, all_paths), total=n))
    
    print('Done.')