"""
CC-RRT

set all angle as rad
no control noise
"""

import math
import os
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import erf

from v1.cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion, draw_vehicle


class CCRRT_SAMPLING(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
    
    def get_random_node(self):
        """
        node sampling
        """
        rnd = self.Node(random.uniform(self.min_rand_x, self.max_rand_x),
                        random.uniform(self.min_rand_y, self.max_rand_y),
                        np.deg2rad(random.uniform(-180, 180)))
        return rnd


def main():
    print("Start " + __file__)

    area = [0, 18, 0, 18]

    # Set Initial parameters
    start = [7.5, 0.0, np.deg2rad(90.0)]
    goal = [7.5, 18.0, np.deg2rad(90.0)]
    car = Vehicle()

    # ====Search Path with CCRRT====
    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    # axis = length + sigma
    # obstacle_list_gt = [
    #     (4, 4, 6, 2, np.deg2rad(90.0)),
    #     (14, 4, 6, 2, np.deg2rad(0.0)),
    #     (14, 14, 6, 2, np.deg2rad(30.0)),
    #     (4, 14, 6, 2, np.deg2rad(60.0)),
    # ]

    # # sigam_ver, sigma_hor, sigma_radius
    # obstacle_list_uncertainty = [
    #     (0.05, 0.02, 0.02),
    #     (0.2, 0.1, 0.06),
    #     (0.5, 0.35, 0.1),
    #     (0.37, 0.2, 0.06),
    # ]

    obstacle_list_gt = [
        (4, 4, 3, 2, np.deg2rad(80.0)),
        # (3, 7, 3, 2, np.deg2rad(65.0)),
        (12, 8, 4, 2.5, np.deg2rad(75.0)),
        # (9, 11, 4, 2, np.deg2rad(80.0)),
        (11, 12, 5, 2.5, np.deg2rad(90.0)),
        # (9, 5, 5, 2.3, np.deg2rad(68.0)),
        (8, 14, 3, 2, np.deg2rad(75.0)),
        # (6, 12, 5, 3, np.deg2rad(80.0)),
    ]

    # sigam_ver, sigma_hor, sigma_radius
    obstacle_list_uncertainty = [
        (0.05, 0.02, 0.02),
        # (0.07, 0.03, 0.02),
        (0.2, 0.1, 0.06),
        # (0.18, 0.15, 0.04),
        (0.5, 0.35, 0.1),
        # (0.4, 0.2, 0.07),
        (0.4, 0.22, 0.07),
        # (0.37, 0.2, 0.06),
    ]

    # area = [0.0, 14.0, 0.0, 50.0]

    # Set Initial parameters
    # start = [12.25, 2.0, np.deg2rad(90.0)]
    # goal = [1.75, 47.0, np.deg2rad(90.0)]
    # car = Vehicle()

    # ====Search Path with CCRRT====
    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    # axis = length + sigma
    # obstacle_list_gt = [
    #     (5.20, 9.0, 5.5, 2.0, np.deg2rad(90.0)),
    #     (1.85, 12.5, 5.2, 2.0, np.deg2rad(87.0)),
    #     (12.25, 11.0, 4.6, 1.8, np.deg2rad(85.0)),
    #     (5.30, 18.0, 5.5, 2.2, np.deg2rad(90.0)),
    #     (8.80, 24.0, 5.5, 2.0, np.deg2rad(90.0)),
    #     (1.75, 23.5, 5.3, 2.1, np.deg2rad(95.0)),
    #     (5.40, 35.5, 5.2, 2.0, np.deg2rad(85.0)),
    #     (12.25, 40.5, 5.3, 2.2, np.deg2rad(95.0)),
    # ]

    # obstacle_list_gt = [
    #     (5.20, 9.0, 5.5, 2.0, np.deg2rad(40.0)),
    #     (1.85, 12.5, 5.2, 2.0, np.deg2rad(175.0)),
    #     (12.25, 11.0, 4.6, 1.8, np.deg2rad(85.0)),
    #     (5.30, 18.0, 5.5, 2.2, np.deg2rad(90.0)),
    #     (8.80, 24.0, 5.5, 2.0, np.deg2rad(120.0)),
    #     (1.75, 23.5, 5.3, 2.1, np.deg2rad(95.0)),
    #     (5.40, 35.5, 5.2, 2.0, np.deg2rad(0.0)),
    #     (12.25, 40.5, 5.3, 2.2, np.deg2rad(95.0)),
    # ]

    # sigam_ver, sigma_hor, sigma_radius
    # obstacle_list_uncertainty = [
    #     (0.17, 0.08, 0.02),
    #     (0.22, 0.10, 0.02),
    #     (0.28, 0.12, 0.03),
    #     (0.34, 0.15, 0.03),
    #     (0.52, 0.40, 0.06),
    #     (0.56, 0.35, 0.06),
    #     (0.72, 0.53, 0.07),
    #     (0.65, 0.51, 0.07),
    # ]

    # (x, y, long_axis, short_axis, radius [-pi, pi])
    # vehicle_length = long_axis * 2
    # vehicle_width = short_axis * 2
    obstacle_list = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)

    # Set Initial parameters
    cc_rrt = CCRRT_SAMPLING(
        car=car,
        start=start,
        goal=goal,
        rand_area=area,
        obstacle_list=obstacle_list)
    cc_rrt.p_safe = 0.95
    t_node = []
    f_node = []
    print(cc_rrt.check_chance_constrain(cc_rrt.start, cc_rrt.p_safe))
    print(cc_rrt.check_chance_constrain(cc_rrt.end, cc_rrt.p_safe))

    if cc_rrt.check_chance_constrain(cc_rrt.start, cc_rrt.p_safe):
        t_node.append(cc_rrt.start)
    else:
        f_node.append(cc_rrt.start)
    
    if cc_rrt.check_chance_constrain(cc_rrt.end, cc_rrt.p_safe):
        t_node.append(cc_rrt.end)
    else:
        f_node.append(cc_rrt.end)

    for _ in range(5000):
        node = cc_rrt.get_random_node()
        # node.yaw = np.deg2rad(90)
        if cc_rrt.check_chance_constrain(node, cc_rrt.p_safe):
            t_node.append(node)
        else:
            f_node.append(node)
    
    cc_rrt.draw_graph()
    draw_vehicle(obstacle_list_gt)

    for node in t_node:
        plt.plot(node.x, node.y, "*b")
        cc_rrt.plot_arrow(node.x, node.y, node.yaw, fc='b')
    
    for node in f_node:
        plt.plot(node.x, node.y, "*y")
        cc_rrt.plot_arrow(node.x, node.y, node.yaw, fc='y')

    plt.figure(2)
    total_node = []
    total_node.extend(t_node)
    total_node.extend(f_node)
    total_cc = [cc_rrt.get_chance_constrain(node) for node in total_node]
    
    plt.scatter([node.x for node in total_node], [node.y for node in total_node], s=3, c=total_cc, cmap='jet')
    plt.colorbar()

    plt.axis([area[0], area[1], area[2], area[3]])
    plt.show()

if __name__ == '__main__':
    main()
