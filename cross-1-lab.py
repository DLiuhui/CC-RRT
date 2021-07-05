import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3'))

from v3.cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion
from v3.cl_rrt import CLRRT

dis_threshold=20
def un_generate(dis, p1, p2):
    base = dis / dis_threshold
    sigma_base = np.abs(np.random.normal(0.0, base * p1))
    #print(sigma_base)
    return (base + sigma_base) * p2

def draw_ground_true(obs_list):
    for obs in obs_list:
       # x=0
        w = obs[3] / 2.0
        l = obs[2] / 2.0
        x = obs[0]
        y = obs[1]
        yaw = obs[4]
        p0 = [
            x + l * math.cos(yaw) + w * math.sin(yaw),
            y + l * math.sin(yaw) - w * math.cos(yaw)
        ]
        p1 = [
            x + l * math.cos(yaw) - w * math.sin(yaw),
            y + l * math.sin(yaw) + w * math.cos(yaw)
        ]
        p2 = [
            x - l * math.cos(yaw) - w * math.sin(yaw),
            y - l * math.sin(yaw) + w * math.cos(yaw)
        ]
        p3 = [
            x - l * math.cos(yaw) + w * math.sin(yaw),
            y - l * math.sin(yaw) - w * math.cos(yaw)
        ]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color="green", lw='1')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="green", lw='1')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color="green", lw='1')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], color="green", lw='1')

def draw_vehicle(obs_list_gt):
    for obs in obs_list_gt:
        w = obs[3] / 2.0
        l = obs[2] / 2.0
        x = obs[0]
        y = obs[1]
        yaw = obs[4]
        p0 = [
            x + l * math.cos(yaw) + w * math.sin(yaw),
            y + l * math.sin(yaw) - w * math.cos(yaw)
        ]
        p1 = [
            x + l * math.cos(yaw) - w * math.sin(yaw),
            y + l * math.sin(yaw) + w * math.cos(yaw)
        ]
        p2 = [
            x - l * math.cos(yaw) - w * math.sin(yaw),
            y - l * math.sin(yaw) + w * math.cos(yaw)
        ]
        p3 = [
            x - l * math.cos(yaw) + w * math.sin(yaw),
            y - l * math.sin(yaw) - w * math.cos(yaw)
        ]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'r')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'r')

def draw_carsize_of_final_path(vehicle, obs_list):
    w = vehicle.w / 2.0
    for obs in obs_list:
        x = obs.x
        y = obs.y
        yaw = obs.yaw
        p0 = [
            x + vehicle.l_f * math.cos(yaw) + w * math.sin(yaw),
            y + vehicle.l_f * math.sin(yaw) - w * math.cos(yaw)
        ]
        p1 = [
            x + vehicle.l_f * math.cos(yaw) - w * math.sin(yaw),
            y + vehicle.l_f * math.sin(yaw) + w * math.cos(yaw)
        ]
        p2 = [
            x - vehicle.l_r * math.cos(yaw) - w * math.sin(yaw),
            y - vehicle.l_r * math.sin(yaw) + w * math.cos(yaw)
        ]
        p3 = [
            x - vehicle.l_r * math.cos(yaw) + w * math.sin(yaw),
            y - vehicle.l_r * math.sin(yaw) - w * math.cos(yaw)
        ]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'k')

def main():
    car = Vehicle()
    car.l_f = 4.51 / 2.0
    car.l_r = 4.51 / 2.0
    car.w = 2.0

    # Set Initial parameters
    start = [-1.75, 13.0, -np.pi/2]
    goal = [4.5, -4, np.deg2rad(90.0)]

    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    obstacle_list_gt = [
        (1.361, 1.348, 3.53, 1.59, 2.237),
        (-5.181, -8.498, 3.64, 1.61, 1.616),
        (1.6467, -8.446, 4.377, 1.692, 1.5526),
        (11.0285, -1.6743, 4.309, 1.645, 3.109),
        (6.3373, -9.1127, 4.0051, 1.585, 1.36976),
        (-5.3691, 9.0414, 3.3907, 1.5249, 1.5587)
    ]
    # (x, y, long_axis, short_axis, radius [-pi, pi], sigma_ver, sigma_hor, l/2, w/2)
    obstacle_list = [
        (1.361, 1.348, 2.36, 1.24, 2.237, 2.36 - 3.53/2, 1.24 - 1.59/2, 3.53/2, 1.59/2),
        (-5.181, -8.498, 2.41, 1.176, 1.616, 2.41 - 3.64/2, 1.176 - 1.61/2, 3.64/2, 1.61/2),
        (1.6467, -8.446, 2.945, 1.259, 1.5526, 2.945 - 4.377/2, 1.259 - 1.692/2, 4.377/2, 1.692/2),
        (11.0285, -1.6743, 2.899, 1.318, 3.109, 2.899 - 4.309/2, 1.318 - 1.645/2, 4.309/2, 1.645/2),
        (6.3373, -9.1127, 2.928, 1.377, 1.36976, 2.928 - 4.0051/2, 1.377-1.585/2, 4.0051/2, 1.585/2),
        (-5.3691, 9.0414, 2.2515, 1.1159, 1.5587, 2.2515 - 3.3907/2, 1.1159 - 1.5249/2, 3.3907/2, 1.5249/2)
    ]

    ground_truth = [
        (-5.3, -8.5, 4.8557, 2.0323, -np.pi/2),
        (-5.3, 9.0, 4.7175, 1.895, -np.pi/2),
        (1.8, -8.5, 4.611, 2.2417, np.pi/2),
        (1.4, 1.2, 4.974, 2.0384, -0.890120),
        (6.4, -9.3, 3.8058, 1.9703, 1.20428),
        (11.0, -1.8, 3.9877, 1.851, 0.0),
        (20.0, -5.3, 3.9877, 1.851, 0.0),
        # (-1.75, 13.0, 4.5135, 2.0068, -np.pi/2)
    ]

    obstacle_list_uncertainty = []
    for obs in obstacle_list_gt:
        dist = np.hypot(start[0] - obs[0], start[1] - obs[1])
        un = (un_generate(dist, 0.5, 0.7),  # sigma_ver
              un_generate(dist, 0.3, 0.6),  # sigma_hor
              un_generate(dist, 0.1, 0.1)  # sigma_radius
              )
        obstacle_list_uncertainty.append(un)
    obstacle_list_for_cc = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)

    obs_gt = []
    for gt in ground_truth:
        obs_gt.append((gt[0], gt[1], gt[2]/1.6, gt[3]/1.6, gt[4], 0.0, 0.0, gt[2]/2.0, gt[3]/2.0))

    area = [-7, 7, -7, 16]  # x-min x-max y-min y-max

    max_cc = []
    avg_cc = []
    avg_wp = []
    avg_len = []
    succ_1 = 0
    succ_2 = 0
    succ_3 = 0

    # 评估用
    cc_gt = CCRRT(car=car, start=start, goal=goal, rand_area=area, obstacle_list=obs_gt) # 与真实框计算碰撞
    cc_gt.p_safe = 0.99
    cc_pu = CCRRT(car=car, start=start, goal=goal, rand_area=area, obstacle_list=obstacle_list) # 使用感知结果结算cc
    cc_pu.p_safe = 0.99

    for i in range(100):
        cc_rrt = CCRRT(
            car=car,
            start=start,
            goal=goal,
            rand_area=area,
            obstacle_list=obstacle_list_for_cc)
        cc_rrt.p_safe = 0.99
        cc_rrt.max_n_node = 2000
        cc_rrt.draw_tree = False
        cc_rrt.planning(animation=False)
        
        valid = True
        for node in cc_rrt.path:
            if not cc_gt.collision_checking(node):
                valid = False
                break
        succ_1 += valid

        valid_2 = False
        if valid and cc_rrt.calc_distance(cc_rrt.path[0], cc_rrt.Node(goal[0], goal[1], goal[2])) < cc_rrt.dis_threshold:
            valid_2 = True
        succ_2 += valid_2

        wp_cc = [cc_pu.get_chance_constrain(node) for node in cc_rrt.path]
        max_cc_ = np.max(wp_cc)
        avg_cc_ = np.mean(wp_cc)

        if valid and valid_2 and max_cc_ < 1 - cc_pu.p_safe:
            succ_3 += 1
        
        max_cc.append(max_cc_)
        avg_cc.append(avg_cc_)
        if valid_2:
            avg_wp.append(float(len(wp_cc)))
            dis = 0
            for j in range(1, len(cc_rrt.path)):
                dis += cc_rrt.calc_distance(cc_rrt.path[j - 1], cc_rrt.path[j])
            avg_len.append(dis)
        
    print("succ-1", succ_1)
    print("succ-2", succ_2)
    print("succ-3", succ_3)
    print("max-cc", np.mean(max_cc))
    print("avg-cc", np.mean(avg_cc))
    print("avg-wp", np.mean(avg_wp))
    print("avg-len", np.mean(avg_len))

if __name__ == "__main__":
    main()

