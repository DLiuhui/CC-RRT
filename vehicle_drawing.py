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
from scipy.special import erf, erfinv
from v1.cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion

class CCRRT_DRAW_VEHICLE(CCRRT):
    
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.goal_sample_rate = 0
    
    def get_random_node(self):
        """
        node sampling
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand_x, self.max_rand_y),
                            random.uniform(self.min_rand_x, self.max_rand_y),
                            np.deg2rad(random.uniform(-180, 180)))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, 0.0)
        return rnd
    
    def draw_car(self, x, y, yaw):
        """
        calculate vehicle's edge constraints
        return ([a_1, a_2, a_3, a_4], [b1, b2, b3, b4])
        """ 
        w = self.car.w / 2.0
        p0 = [
            x + self.car.l_f * math.cos(yaw) + w * math.sin(yaw),
            y + self.car.l_f * math.sin(yaw) - w * math.cos(yaw)
        ]
        p1 = [
            x + self.car.l_f * math.cos(yaw) - w * math.sin(yaw),
            y + self.car.l_f * math.sin(yaw) + w * math.cos(yaw)
        ]
        p2 = [
            x - self.car.l_r * math.cos(yaw) - w * math.sin(yaw),
            y - self.car.l_r * math.sin(yaw) + w * math.cos(yaw)
        ]
        p3 = [
            x - self.car.l_r * math.cos(yaw) + w * math.sin(yaw),
            y - self.car.l_r * math.sin(yaw) - w * math.cos(yaw)
        ]
        # calculate k and b
        # kx + b = y => kx - y + b = 0
        # k = (y1-y2)/(x1-x2) b = (y2x1-y1x2)/(x1-x2)
        k = [p0[0] - p3[0], p0[1] - p3[1]] # 单位外法向量
        d = math.sqrt(k[0]**2 + k[1]**2)
        k = [i / d for i in k]
        s = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0]
        e = [s[0] + k[0], s[1] + k[1]]
        plt.plot([s[0], e[0]], [s[1], e[1]], 'y')
        plt.plot(e[0], e[1], '*y')

        k = [p1[0] - p0[0], p1[1] - p0[1]] # 单位外法向量
        d = math.sqrt(k[0]**2 + k[1]**2)
        k = [i / d for i in k]
        s = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
        e = [s[0] + k[0], s[1] + k[1]]
        plt.plot([s[0], e[0]], [s[1], e[1]], 'c')
        plt.plot(e[0], e[1], '*c')

        k = [p2[0] - p1[0], p2[1] - p1[1]] # 单位外法向量
        d = math.sqrt(k[0]**2 + k[1]**2)
        k = [i / d for i in k]
        s = [(p2[0] + p3[0]) / 2.0, (p2[1] + p3[1]) / 2.0]
        e = [s[0] + k[0], s[1] + k[1]]
        plt.plot([s[0], e[0]], [s[1], e[1]], 'g')
        plt.plot(e[0], e[1], '*g')

        k = [p3[0] - p2[0], p3[1] - p2[1]] # 单位外法向量
        d = math.sqrt(k[0]**2 + k[1]**2)
        k = [i / d for i in k]
        s = [(p3[0] + p0[0]) / 2.0, (p3[1] + p0[1]) / 2.0]
        e = [s[0] + k[0], s[1] + k[1]]
        plt.plot([s[0], e[0]], [s[1], e[1]], 'm')
        plt.plot(e[0], e[1], '*m')

        plt.scatter(x, y, c='r', marker='X')
        self.plot_arrow(x, y, yaw, fc='r')

        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'k')


def main():
    print("Start " + __file__)

    area = [-2, 20, -2, 20]

    # Set Initial parameters
    start = [7.5, 0.0, np.deg2rad(90.0)]
    goal = [7.5, 18.0, np.deg2rad(90.0)]
    car = Vehicle()

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

    # sigam_ver, sigma_hor, sigma_radius
    obstacle_list_uncertainty = [
        (0.05, 0.02, 0.02),
        (0.07, 0.03, 0.02),
        (0.2, 0.1, 0.06),
        (0.18, 0.15, 0.04),
        (0.5, 0.35, 0.1),
        (0.4, 0.2, 0.07),
        (0.4, 0.22, 0.07),
        (0.37, 0.2, 0.06),
    ]

    # (x, y, long_axis, short_axis, radius [-pi, pi])
    # vehicle_length = long_axis * 2
    # vehicle_width = short_axis * 2
    obstacle_list = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)

    # Set Initial parameters
    cc_rrt = CCRRT_DRAW_VEHICLE(
        car=car,
        start=start,
        goal=goal,
        rand_area=area,
        obstacle_list=obstacle_list)
    node = cc_rrt.get_random_node()
    cc_rrt.draw_graph()
    # cc_rrt.draw_car(cc_rrt.start.x, cc_rrt.start.y, cc_rrt.start.yaw)
    # cc_rrt.draw_car(cc_rrt.end.x, cc_rrt.end.y, cc_rrt.end.yaw)
    cc_rrt.draw_car(node.x, node.y, node.yaw)
    plt.show()

if __name__ == '__main__':
    main()
