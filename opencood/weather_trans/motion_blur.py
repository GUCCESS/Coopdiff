import numpy as np

def apply_motion_blur_to_numpy(points):


    np.random.seed(420)


    trans_std = 0.2 #0.2
    # Add Gaussian noise to each point to simulate motion blur
    noise_translate = np.random.normal(0, trans_std, points.shape)
    blurred_points = points + noise_translate
    return blurred_points
