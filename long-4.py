import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2'))

from v2.cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion
from v2.cl_rrt import CLRRT

dis_threshold=25.5
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
    start = [9.066, 67.66, np.deg2rad(90.0)]
    goal = [2.0, 100.0, np.deg2rad(90.0)]

    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    obstacle_list_gt = [
        (1.929, 74.62, 4.29, 1.6, np.deg2rad(90.0)),
        # (5.36, 44.96, 3.96, 1.59, np.deg2rad(100.0),2.61, 1.19),
        # (5.44, 65.03, 3.56, 1.54, np.deg2rad(89.0), 2.34, 1.11),
        (5.56, 94.53, 4.31, 1.66, np.deg2rad(90.0)),
        # (9.0, 54.93, 4.08, 1.59, np.deg2rad(90.0), 2.6, 1.14),
        (9.128, 84.93, 3.62, 1.64, np.deg2rad(90.0)),
        # (12.73, 44.84, 3.4, 1.5, np.deg2rad(89.90), 2.28, 1.13),
        # (12.55, 64.96, 4.45, 1.71, np.deg2rad(90.0), 2.83, 1.18),
    ]
    # (x, y, long_axis, short_axis, radius [-pi, pi], sigma_ver, sigma_hor, l/2, w/2)
    obstacle_list = [
        (1.929, 74.62, 2.89, 1.21, np.deg2rad(90.0), 2.89 - 4.29/2, 1.21 - 1.6/2, 4.29/2, 1.6/2),
        # (5.36, 44.96, 3.96, 1.59, np.deg2rad(100.0),2.61, 1.19),
        # (5.44, 65.03, 3.56, 1.54, np.deg2rad(89.0), 2.34, 1.11),
        (5.56, 94.53, 2.96, 1.27, np.deg2rad(90.0), 2.96 - 4.31/2, 1.27 - 1.66/2, 4.31/2, 1.66/2),
        # (9.0, 54.93, 4.08, 1.59, np.deg2rad(90.0), 2.6, 1.14),
        (9.128, 84.93, 2.32, 1.14, np.deg2rad(90.0), 2.32 - 3.62/2, 1.14 - 1.64/2, 3.62/2, 1.64/2),
        # (12.73, 44.84, 3.4, 1.5, np.deg2rad(89.90), 2.28, 1.13),
        # (12.55, 64.96, 4.45, 1.71, np.deg2rad(90.0), 2.83, 1.18),
    ]

    ground_truth = [
        (1.9, 75.0, 4.72, 1.89, np.deg2rad(90.0)),
        (5.5, 45.0, 4.19, 1.82, np.deg2rad(90.0)),
        (5.5, 65.0, 4.79, 2.16, np.deg2rad(90.0)),
        (5.5, 95.0, 5.36, 2.03, np.deg2rad(90.0)),
        (9.1, 55.0, 4.86, 2.03, np.deg2rad(90.0)),
        (9.1, 85.0, 3.99, 1.85, np.deg2rad(90.0)),
        (12.7, 45.0, 4.18, 1.99, np.deg2rad(90.0)),
        (12.7, 65.0, 4.61, 2.24, np.deg2rad(90.0)),
    ]

    obstacle_list_uncertainty = []
    for obs in obstacle_list_gt:
        dist = np.hypot(start[0] - obs[0], start[1] - obs[1])
        un = (un_generate(dist, 0.4, 0.7),  # sigma_ver
              un_generate(dist, 0.25, 0.6),  # sigma_hor
              un_generate(dist, 0.15, 0.05)  # sigma_radius
              )
        obstacle_list_uncertainty.append(un)
    obstacle_list_for_cc = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)

    area = [0, 15, 65, 100]  # x-min x-max y-min y-max
    cc_rrt = CLRRT(
        car=car,
        start=start,
        goal=goal,
        rand_area=area,
        obstacle_list=obstacle_list)
    cc_rrt.p_safe = 0.99
    cc_rrt.max_n_node = 5000
    cc_rrt.draw_tree = False
    cc_rrt.planning(animation=False)
    
    area = [0, 15, 30, 100]  # x-min x-max y-min y-max
    plt.figure(1, figsize=(4.5, 9.5))
    cc_rrt.draw_graph()
    cc_rrt.draw_path()
    draw_vehicle(obstacle_list_gt)
    draw_ground_true(ground_truth)
    draw_carsize_of_final_path(car, cc_rrt.path)
    plt.axis("equal")
    plt.axis([area[0], area[1], area[2], area[3]])
    # 画路
    plt.plot([0, 0], [30, 105], color="grey")
    plt.plot([14.4, 14.4], [30, 105], color="grey")
    plt.plot([3.6, 3.6], [30, 105], "--", color="grey")
    plt.plot([7.2, 7.2], [30, 105], "--", color="grey")
    plt.plot([10.8, 10.8], [30, 105], "--", color="grey")

    plt.figure(2, figsize=(4.5, 9.5))
    cc_rrt.draw_graph()
    draw_vehicle(obstacle_list_gt)
    draw_ground_true(ground_truth)
    tmp = [node.cc for node in cc_rrt.path]  # 从这个可以看出这个finalpath里面都是一些节点
    # print(tmp)
    print(len(cc_rrt.path))
    path_min = np.min(tmp)
    path_max = np.max(tmp)
    path_avg = np.average(tmp)
    
    # plt.axes([0.3, 0.1, 8 / 50, 8 / 10.55])
    plt.scatter([node.x for node in cc_rrt.node_list],
                [node.y for node in cc_rrt.node_list],
                s=3,
                c=[node.cc for node in cc_rrt.node_list],
                cmap='jet')
    plt.plot([node.x for node in cc_rrt.path],
             [node.y for node in cc_rrt.path],
             c='k',
             label="path risk value:\nmin: %.6f\nmax: %.6f\navg: %.6f" % (path_min, path_max, path_avg))
    plt.colorbar()
    plt.axis("equal")
    plt.axis([area[0], area[1], area[2], area[3]])
    # 画路
    plt.plot([0, 0], [30, 105], color="grey")
    plt.plot([14.4, 14.4], [30, 105], color="grey")
    plt.plot([3.6, 3.6], [30, 105], "--", color="grey")
    plt.plot([7.2, 7.2], [30, 105], "--", color="grey")
    plt.plot([10.8, 10.8], [30, 105], "--", color="grey")

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

