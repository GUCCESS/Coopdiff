import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress


def frenel_equations_power(ain, nair=1.0003, nw=1.33):
    """
    Calculate the power ratio of transmitted laser power
    """
    a = np.clip(np.sin(ain) * nair / nw, -1, 1)
    aout = np.arcsin(a)

    power_fraction_transmittance = np.cos(ain) * nair / nw / np.cos(aout)

    rs = (nair * np.cos(ain) - nw * np.cos(aout)) / (nair * np.cos(ain) + nw * np.cos(aout))
    ts = 2 * nair * np.cos(ain) / (nair * np.cos(ain) + nw * np.cos(aout))

    rp = (nw * np.cos(ain) - nair * np.cos(aout)) / (nw * np.cos(ain) + nair * np.cos(aout))
    tp = 2 * nair * np.cos(ain) / (nw * np.cos(ain) + nair * np.cos(aout))

    rs = rs ** 2
    ts = ts ** 2 / power_fraction_transmittance

    rp = rp ** 2
    tp = tp ** 2 / power_fraction_transmittance

    return rs, ts, rp, tp, aout


def total_reflection_from_ground(ain, nair=1.0003, nw=1.33, rho=0.9):
    """
    Defines optical transition from air to water
    """
    ras, tas, rap, tap, aaout = frenel_equations_power(ain, nair=nair, nw=nw)
    rws, tws, rwp, twp, awout = frenel_equations_power(aaout, nair=nw, nw=nair)

    rs = ras
    ts = tas * rho * tws / (1 - rho * rws)

    rp = rap
    tp = tap * rho * twp / (1 - rho * rwp)

    return rs, ts, rp, tp, aaout


def total_transmittance_from_ground(ain, nair=1.0003, nw=1.33, rho=0.9, water_absorbtion=0.075, water_thickness=0.0025):
    """
    Defines power transmission for air-water transition and backprojection
    """
    rs, ts, rp, tp, aaout = total_reflection_from_ground(ain, nair=nair, nw=nw, rho=rho)
    return rs, ts, rp, tp, aaout


def ransac_polyfit(x, y, order=3, n=15, k=100, t=0.1, d=15, f=0.8):
    """
    RANSAC-based polynomial fitting
    """
    bestfit = np.polyfit(x, y, order)
    besterr = np.sum(np.abs(np.polyval(bestfit, x) - y))
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit

def estimate_laser_parameters(pointcloud_planes, calculated_indicent_angle, power_factor=15, noise_floor=0.7,
                              debug=True, estimation_method='linear'):
    """
    Estimate laser output power and noise level parameters
    """
    normalized_intensitites = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle)
    distance = np.linalg.norm(pointcloud_planes[:, :3], axis=1)

    p = None
    stat_values = None
    if len(normalized_intensitites) < 3:
        return None, None, None, None
    
    if estimation_method == 'linear':
        reg = linregress(distance, normalized_intensitites)
        w = reg[0]
        h = reg[1]
        p = [w, h]
        stat_values = reg[2:]
        relative_output_intensity = power_factor * (p[0] * distance + p[1])
    elif estimation_method == 'poly':
        p = np.polyfit(distance, normalized_intensitites, 2)
        relative_output_intensity = power_factor * (p[0] * distance **2 + p[1] * distance + p[2])

    hist, xedges, yedges = np.histogram2d(
        distance, normalized_intensitites,
        bins=(50, 2555),
        range=((10, 70), (0, np.abs(np.max(normalized_intensitites))))
    )
    
    # --- 修复部分开始 ---
    # 找到直方图中值为0的索引
    idx_hist_zero = np.where(hist == 0)
    # 使用元组索引方式进行赋值，避免FutureWarning
    hist[idx_hist_zero] = len(pointcloud_planes)
    
    # 找到每行中第二小的强度值对应的索引
    ymins = np.argpartition(hist, 2, axis=1)[:, 0]
    # 使用这些索引从yedges中获取实际的强度值，这正是min_vals的来源
    min_vals = yedges[ymins]
    
    # 找到min_vals中大于5的索引
    idx_min_vals = np.where(min_vals > 5)[0]
    min_vals = min_vals[idx_min_vals]
    
    # 根据idx_min_vals，获取对应的x轴（距离）值
    x = (xedges[idx_min_vals] + xedges[idx_min_vals + 1]) / 2
    # --- 修复部分结束 ---

    if estimation_method == 'poly':
        pmin = ransac_polyfit(x, min_vals, order=2)
        adaptive_noise_threshold = noise_floor * (pmin[0] * distance** 2 + pmin[1] * distance + pmin[2])
    elif estimation_method == 'linear':
        if len(min_vals) > 3:
            pmin = linregress(x, min_vals)
        else:
            pmin = p
        adaptive_noise_threshold = noise_floor * (pmin[0] * distance + pmin[1])

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(distance, normalized_intensitites, 'x')
        plt.plot(distance, relative_output_intensity, 'x')
        plt.plot(distance, adaptive_noise_threshold, 'x')
        plt.title('Estimated Lidar Parameters')
        plt.ylabel('Intensity')
        plt.xlabel('distance')
        plt.legend(['Input Intensities', 'Total Power', 'Noise Level'])
        plt.show()

    return relative_output_intensity, adaptive_noise_threshold, p, stat_values


def calculate_plane(pointcloud, standart_height=-1.55):
    """
    Calculate ground plane using RANSAC regression
    """
    valid_loc = (pointcloud[:, 2] < -1.55) & \
                (pointcloud[:, 2] > -1.86 - 0.01 * pointcloud[:, 0]) & \
                (pointcloud[:, 0] > 10) & \
                (pointcloud[:, 0] < 70) & \
                (pointcloud[:, 1] > -3) & \
                (pointcloud[:, 1] < 3)
                
    pc_rect = pointcloud[valid_loc]

    if pc_rect.shape[0] <= pc_rect.shape[1]:
        w = [0, 0, 1]
        h = standart_height
    else:
        try:
            reg = RANSACRegressor(loss='squared_error', max_trials=1000).fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[1] = reg.estimator_.coef_[1]
            w[2] = -1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)
        except:
            print('Was not able to estimate a ground plane. Using default flat earth assumption')
            w = [0, 0, 1]
            h = standart_height

    return w, h


def ground_water_augmentation(points, water_height=0.0012, pavement_depth=0.0012, noise_floor=0.25, 
                              power_factor=15, estimation_method='linear', flat_earth=True, 
                              debug=False, delta=0.7, replace=False):
    """
    仅处理点云数据的地面潮湿增强函数，输入输出均为点云数组
    
    :param points: 点云数据，格式为N×5的numpy数组 (x, y, z, intensity, channel)
    :return: 增强后的点云数据，格式保持N×5的numpy数组
    """
    # :param points: 点云数据，格式为N×5的numpy数组 (x, y, z, intensity, channel)
    # :param water_height: 水层的高度（单位：米）。
    # :param pavement_depth: 路面/地面的深度（单位：米），用于计算水层覆盖的比例。
    # :param noise_floor: 噪声阈值因子，用于估计自适应噪声水平。
    # :param power_factor: 激光输出功率的缩放因子。
    # :param estimation_method: 估计激光参数的方法，'linear'（线性回归）或'poly'（多项式拟合）。
    # :param flat_earth: 是否假设地面是完全平坦的，这将影响入射角的计算。
    # :param debug: 是否开启调试模式，如果为True，将显示参数估计的图表。
    # :param delta: 用于筛选地面点的距离阈值（单位：米）。
    # :param replace: 是否将所有地面点的channel值替换为0。
    # :return: 增强后的点云数据，格式保持N×5的numpy数组

    np.random.seed(420)
    water_height = 0.0015 #0.0005 0.001 0.0015  0.0015(delta0.7 之前的0.5)
    delta = 0.5  #0.5  0.7

    # 平面拟合识别地面点
    w, h = calculate_plane(points)
    height_over_ground = np.matmul(points[:, :3], np.asarray(w)).reshape((-1, 1))
    # 根据平面距离筛选地面点
    ground = np.logical_and(
        np.matmul(points[:, :3], np.asarray(w)) + h < delta,
        np.matmul(points[:, :3], np.asarray(w)) + h > -delta
    )
    ground_idx = np.where(ground)
    pointcloud_planes = np.hstack((points[ground, :], height_over_ground[ground]))

    # 若地面点不足，直接返回原始点云
    if pointcloud_planes.shape[0] < 1000:
        return points
    
    # 计算入射角
    if not flat_earth:
        calculated_indicent_angle = np.arccos(
            np.divide(
                np.matmul(pointcloud_planes[:, :3], np.asarray(w)),
                np.linalg.norm(pointcloud_planes[:, :3], axis=1) * np.linalg.norm(w)
            )
        )
    else:
        calculated_indicent_angle = np.arccos(
            -np.divide(
                np.matmul(pointcloud_planes[:, :3], np.asarray([0, 0, 1])),
                np.linalg.norm(pointcloud_planes[:, :3], axis=1) * np.linalg.norm([0, 0, 1])
            )
        )

    # 估计激光参数
    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(
        pointcloud_planes, calculated_indicent_angle,
        noise_floor=noise_floor, estimation_method=estimation_method,
        power_factor=power_factor, debug=debug
    )

    # 计算反射率和透射率
    reflectivities = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle) / relative_output_intensity
    rs, ts, rp, tp, _ = total_transmittance_from_ground(
        calculated_indicent_angle, rho=np.clip(reflectivities, 0.05, 1)
    )
    t = np.maximum(tp, ts)

    # 计算加权反射率（潮湿地面模型）
    f = np.clip(water_height / pavement_depth, 0, 1)
    tw = (1 - f) * reflectivities + f * t / calculated_indicent_angle

    # 调整强度并过滤点
    new_intensities = np.clip(
        relative_output_intensity * np.cos(calculated_indicent_angle) * tw,
        0, pointcloud_planes[:, 3]
    )
    keep_points = new_intensities > adaptive_noise_threshold * np.cos(calculated_indicent_angle)
    keep_points_idx = np.where(keep_points)
    pointcloud_planes = pointcloud_planes[:, :5]  # 保留原始5列数据

    # 拼接结果点云（非地面点 + 过滤后的地面点）
    augmented_points = np.zeros((
        points.shape[0] - ground_idx[0].shape[0] + keep_points_idx[0].shape[0], 4
    ))
    augmented_points[:points.shape[0] - ground_idx[0].shape[0], :] = points[np.logical_not(ground), :]
    augmented_points[points.shape[0] - ground_idx[0].shape[0]:, :] = pointcloud_planes[keep_points_idx][:, :4]
    # 更新地面点强度
    augmented_points[points.shape[0] - ground_idx[0].shape[0]:, 3] = new_intensities[keep_points_idx]

    if replace:
        augmented_points[:, 4] = 0

    return augmented_points