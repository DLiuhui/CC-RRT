import numpy as np
import matplotlib.pyplot as plt

dis_threshold = 25.5

def un_generate(dis, p1, p2):
    base = dis / dis_threshold
    sigma_base = np.abs(np.random.normal(0.0, base * p1))
    print(sigma_base)
    return (base + sigma_base) * p2

def obstacle_uncertainty_fusion(gts, uncertainties):
    obs = []
    for gt, un in zip(gts, uncertainties):
        a = gt[2] / 2.0
        b = gt[3] / 2.0
        d_a = a * (1 - b * np.sqrt((1 + np.tan(un[2])**2) / (b**2 + a**2 * np.tan(un[2])**2)))
        d_b = d_a / a * b
        obs.append((gt[0], gt[1], a + d_a + un[0], b + d_b + un[1], gt[4]))
        # obs.append((gt[0], gt[1], d_a + un[0], d_b + un[1], gt[4]))
    return obs

# 测试用
def main():
    # Set Initial parameters
    start = [7.5, -1.0, np.deg2rad(90.0)]

    # ====Search Path with CCRRT====
    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    # axis = length + sigma
    obstacle_list_gt = [
        (4, 4, 3, 2, np.deg2rad(80.0)),
        (3, 7, 3, 2, np.deg2rad(65.0)),
        (12, 8, 4, 2.5, np.deg2rad(75.0)),
        (9, 11, 4, 2, np.deg2rad(80.0)),
        (11, 12, 5, 2.5, np.deg2rad(90.0)),
        (9, 5, 5, 2.3, np.deg2rad(68.0)),
        (8, 14, 3, 2, np.deg2rad(75.0)),
        (6, 12, 5, 3, np.deg2rad(80.0)),
    ]

    obstacle_list_uncertainty = []
    # 为每个障碍物生成不确定性
    for obs in obstacle_list_gt:
        dist = np.hypot(start[0] - obs[0], start[1]-obs[1])
        un = (un_generate(dist, 0.5, 1.0), # sigma_ver
              un_generate(dist, 0.3, 0.85), # sigma_hor
              un_generate(dist, 0.2, 0.1) # sigma_radius
              )
        obstacle_list_uncertainty.append(un)

    # (x, y, long_axis, short_axis, radius [-pi, pi])
    # vehicle_length = long_axis * 2
    # vehicle_width = short_axis * 2
    obstacle_list = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)
    print(obstacle_list)


if __name__ == "__main__":
    main()
