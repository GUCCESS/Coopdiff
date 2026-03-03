import numpy as np

def get_pcd_ringID(points, vertical_resolution=64):
    scan_x = points[:, 0]
    scan_y = points[:, 1]

    yaw = -np.arctan2(scan_y, -scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    ringID = np.cumsum(proj_y)
    ringID = np.clip(ringID, 0, vertical_resolution - 1)
    return ringID


def apply_beam_missing_to_numpy(points):

    np.random.seed(420)

    num_beam_to_drop = 32 # 8 16 32 48
    vertical_resolution=64
    ringID = get_pcd_ringID(points, vertical_resolution=vertical_resolution)
    ringID = ringID.astype(np.int64)

    drop_range = np.arange(vertical_resolution)
    drop_indices = np.random.choice(drop_range, num_beam_to_drop, replace=False)
    drop_mask = np.isin(ringID, drop_indices)
    remaining_points = points[~drop_mask]

    if remaining_points.shape[0] < 1:
        pick = np.random.choice(points.shape[0], min(1, points.shape[0]), replace=False)
        remaining_points = points[pick]

    return remaining_points


# def apply_beam_missing_to_numpy(points):
#     # 如果输入点云本身点数很少，直接返回
#     if points.shape[0] < 10:
#         return points

#     vertical_resolution = 64
#     ringID = get_pcd_ringID(points, vertical_resolution=vertical_resolution)
#     ringID = ringID.astype(np.int64)

#     # 定义要尝试的破坏等级，从高到低
#     drop_levels = [32, 16, 8]
    
#     drop_range = np.arange(vertical_resolution)

#     for num_beam_to_drop in drop_levels:
#         # 随机选择要丢弃的光束
#         drop_indices = np.random.choice(drop_range, num_beam_to_drop, replace=False)
#         drop_mask = np.isin(ringID, drop_indices)
#         remaining_points = points[~drop_mask]

#         # 如果处理后的点云不为空，则直接返回结果
#         if remaining_points.shape[0] > 0:
#             return remaining_points

#     # 如果所有等级的尝试都导致点云为空，则返回原始点云
#     return points
