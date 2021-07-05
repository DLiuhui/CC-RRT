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

from v1.cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion


class CCRRT_SAMPLING(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
    
    def get_random_node(self):
        """
        node sampling
        """
        rnd = self.Node(random.uniform(self.min_rand_x, self.max_rand_x),
                        random.uniform(self.min_rand_y, self.max_rand_y),
                        # np.deg2rad(random.uniform(-180, 180)))
                        np.deg2rad(0))
        return rnd
    
    def is_obs_free(self, node):
        # discard point in obstacle range
        for obs in self.obstacle_list:
            a = ((node.x - obs[0]) * math.cos(obs[4]) + (node.y - obs[1]) * math.sin(obs[4]))**2 / obs[2]**2
            b = ((obs[0] - node.x) * math.sin(obs[4]) + (node.y - obs[1]) * math.cos(obs[4]))**2 / obs[3]**2
            if a + b <= 1:
                return False
        return True
    
    def draw_car(self, x, y, yaw):
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

        plt.plot(x, y, 'Xr')
        self.plot_arrow(x, y, yaw, fc='r')

        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'k')
    
    def get_chance_constrain_his(self, current):
        A, B = self.vehicle_constraints(current.x, current.y, current.yaw)
        delta_t = 0 # sum(min delte_tj)
        molecular = [] # 分子
        denominator = [] # 分母
        erf_item = [] # erf项
        delte_his = [] # delta
        # cal for each obs
        for obs in self.obstacle_list:
            delta_tj = math.inf
            for a, b in zip(A, B):
                a = np.array([a]) # 1*3
                x = np.array([[obs[0]], [obs[1]], [0.0]]) # 3*1
                sigma = current.conv + np.diag([obs[2], obs[3], obs[4]]) # 3*3
                molecular_ = a.dot(x).item() - b
                denominator_ = 2 * a.dot(sigma).dot(a.transpose()).item()
                erf_item_ = molecular_ / np.sqrt(denominator_)
                cc = 0.5 * (1 - erf(erf_item_))

                molecular.append(molecular_)
                denominator.append(denominator_)
                erf_item.append(erf_item_)
                delte_his.append(cc)

                if cc < delta_tj:
                    delta_tj = cc
            delta_t += delta_tj
        return delta_t, molecular, denominator, erf_item, delte_his, A, B


def main():
    print("Start " + __file__)

    area = [0.0, 14.0, 0.0, 15.0]

    # Set Initial parameters
    start = [12.25, 2.0, np.deg2rad(90.0)]
    goal = [1.75, 15.0, np.deg2rad(90.0)]
    car = Vehicle()
    car.l_f = 2.7
    car.l_r = 2.5
    car.w = 2.0

    # ====Search Path with CCRRT====
    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    # axis = length + sigma
    obstacle_list_gt = [
        (5.20, 9.0, 5.5, 2.0, np.deg2rad(90.0)),
        # (1.85, 12.5, 5.2, 2.0, np.deg2rad(87.0)),
        # (12.25, 11.0, 4.6, 1.8, np.deg2rad(85.0)),
        # (5.30, 18.0, 5.5, 2.2, np.deg2rad(90.0)),
        # (8.80, 24.0, 5.5, 2.0, np.deg2rad(90.0)),
        # (1.75, 23.5, 5.3, 2.1, np.deg2rad(95.0)),
        # (5.40, 35.5, 5.2, 2.0, np.deg2rad(85.0)),
        # (12.25, 40.5, 5.3, 2.2, np.deg2rad(95.0)),
    ]

    # sigam_ver, sigma_hor, sigma_radius
    obstacle_list_uncertainty = [
        (0.17, 0.08, 0.02),
        # (0.22, 0.10, 0.02),
        # (0.28, 0.12, 0.03),
        # (0.34, 0.15, 0.03),
        # (0.52, 0.40, 0.06),
        # (0.56, 0.35, 0.06),
        # (0.72, 0.53, 0.07),
        # (0.65, 0.51, 0.07),
    ]

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
    cc_rrt.p_safe = 0.80

    err_node = []

    if cc_rrt.check_chance_constrain(cc_rrt.start, cc_rrt.p_safe) and not cc_rrt.is_obs_free(cc_rrt.start):
        err_node.append(cc_rrt.start)
    
    if cc_rrt.check_chance_constrain(cc_rrt.end, cc_rrt.p_safe) and not cc_rrt.is_obs_free(cc_rrt.end):
        err_node.append(cc_rrt.end)

    for _ in range(40000):
        node = cc_rrt.get_random_node()

        # node.yaw = np.deg2rad(90)
        if cc_rrt.check_chance_constrain(node, cc_rrt.p_safe) and not cc_rrt.is_obs_free(node):
            node.cc = cc_rrt.get_chance_constrain(node)
            err_node.append(node)

    for node in err_node:
        plt.figure()
        cc_rrt.draw_graph()        
        cc_rrt.draw_car(node.x, node.y, node.yaw)
        delta_t, molecular, denominator, erf_item, delte_his, a, b = cc_rrt.get_chance_constrain_his(node)
        print("="*20)
        print("delta_t:", delta_t)
        print("molecular:", molecular)
        print("denominator:", denominator)
        print("erf_item:", erf_item)
        print("delte_his:", delte_his)
        print("a:", a)
        print("b:", b)
        # plt.axis([area[0], area[1], area[2], area[3]])
        # plt.axis("equal")
        # plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
