"""
metric for planning algo
"""

import math
import os
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from v1.cc_rrt import CCRRT, obstacle_uncertainty_fusion, draw_vehicle, draw_carsize_of_final_path, Vehicle

SAVE_ROOT = "/home/ddrh/Program/PU_RRT/v1/traffic_sim"


def main():
    print("Start " + __file__)

    area = [0.0, 14.0, 0.0, 50.0]

    # Set Initial parameters
    start = [12.25, 2.0, np.deg2rad(90.0)]
    goal = [1.75, 47.0, np.deg2rad(90.0)]
    car = Vehicle()

    # ====Search Path with CCRRT====
    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    # axis = length + sigma
    obstacle_list_gt = [
        (5.20, 9.0, 5.5, 2.0, np.deg2rad(90.0)),
        (1.85, 12.5, 5.2, 2.0, np.deg2rad(87.0)),
        (12.25, 11.0, 4.6, 1.8, np.deg2rad(85.0)),
        (5.30, 18.0, 5.5, 2.2, np.deg2rad(90.0)),
        (8.80, 24.0, 5.5, 2.0, np.deg2rad(90.0)),
        (1.75, 23.5, 5.3, 2.1, np.deg2rad(95.0)),
        (5.40, 35.5, 5.2, 2.0, np.deg2rad(85.0)),
        (12.25, 40.5, 5.3, 2.2, np.deg2rad(95.0)),
    ]

    # sigam_ver, sigma_hor, sigma_radius
    obstacle_list_uncertainty = [
        (0.12, 0.05, 0.02),
        (0.17, 0.07, 0.02),
        (0.23, 0.07, 0.03),
        (0.29, 0.10, 0.03),
        (0.46, 0.35, 0.06),
        (0.51, 0.30, 0.06),
        (0.65, 0.46, 0.07),
        (0.58, 0.45, 0.07),
    ]

    # (x, y, long_axis, short_axis, radius [-pi, pi])
    # vehicle_length = long_axis * 2
    # vehicle_width = short_axis * 2
    obstacle_list = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)
    waiting = 0

    # simulation with constant obstacle 
    # not considered env / perception change 没有考虑环境的变化和感知变化
    max_waiting_step = 5

    final_path = []

    while True:
        print(start)
        # Set Initial parameters
        cc_rrt = CCRRT(
            car=car,
            start=start,
            goal=goal,
            rand_area=area,
            obstacle_list=obstacle_list)
        cc_rrt.p_safe = 0.95 - waiting * 0.01
        # path = cc_rrt.planning(animation=False)
        cc_rrt.planning(animation=False)

        plt.figure(1, figsize=(4.5, 9.5))
        cc_rrt.draw_graph()
        cc_rrt.draw_path()
        draw_vehicle(obstacle_list_gt)
        draw_carsize_of_final_path(car, cc_rrt.path)
        # 画车道线
        plt.plot([0.0, 0.0], [0.0, 50.0], c='gray')
        plt.plot([3.5, 3.5], [0.0, 50.0], c='gray')
        plt.plot([7.0, 7.0], [0.0, 50.0], c='gray')
        plt.plot([10.5, 10.5], [0.0, 50.0], c='gray')
        plt.plot([14.0, 14.0], [0.0, 50.0], c='gray')

        plt.axis([0.0, 14.0, 0.0, 50.0])

        for node in cc_rrt.path:
            final_path.append(node)

        if cc_rrt.calc_distance(cc_rrt.path[0], cc_rrt.end) > cc_rrt.dis_threshold: # not arrive goal
            if cc_rrt.path[0].x == start[0] and cc_rrt.path[0].y == start[1]: # stop and waiting
                waiting = min(waiting + 1, max_waiting_step)
            else:
                waiting = max(waiting - 1, 0)
            start = [cc_rrt.path[0].x, cc_rrt.path[0].y, cc_rrt.path[0].yaw] # update start
            area[2] = cc_rrt.path[0].y - 3

        else:
            break
        
        plt.show()
    
    # draw final path
    plt.figure(1, figsize=(4.5, 9.5))
    cc_rrt.draw_graph()
    cc_rrt.draw_path()
    draw_vehicle(obstacle_list_gt)
    draw_carsize_of_final_path(car, cc_rrt.path)
    # 画车道线
    plt.plot([0.0, 0.0], [0.0, 50.0], c='gray')
    plt.plot([3.5, 3.5], [0.0, 50.0], c='gray')
    plt.plot([7.0, 7.0], [0.0, 50.0], c='gray')
    plt.plot([10.5, 10.5], [0.0, 50.0], c='gray')
    plt.plot([14.0, 14.0], [0.0, 50.0], c='gray')
    plt.axis([0.0, 14.0, 0.0, 50.0])

    # if len(cc_rrt.path):
    #     plt.figure(2, figsize=(4.0, 8.0))
    #     tmp = [node.cc for node in cc_rrt.path]
    #     path_min = np.min(tmp)
    #     path_max = np.max(tmp)
    #     path_avg = np.average(tmp)
    #     plt.scatter([node.x for node in cc_rrt.node_list], 
    #                 [node.y for node in cc_rrt.node_list], 
    #                 s=3, 
    #                 c=[node.cc for node in cc_rrt.node_list], 
    #                 cmap='jet')
    #     plt.plot([node.x for node in cc_rrt.path],
    #             [node.y for node in cc_rrt.path],
    #             c='k',
    #             label="path risk value:\nmin: %.3f\nmax: %.3f\navg: %.3f"%(path_min, path_max, path_avg))
    #     plt.colorbar()
    #     plt.axis([0.0, 14.0, 0.0, 50.0])
    #     plt.legend(loc='upper right')
    #     plt.grid(True)
    
    plt.figure(2, figsize=(4.5, 9.5))
    tmp = [node.cc for node in final_path]
    path_min = np.min(tmp)
    path_max = np.max(tmp)
    path_avg = np.average(tmp)
    
    # for node in final_path:
    #     cc_rrt.plot_arrow(node.x, node.y, node.yaw, fc='k')

    plt.scatter([node.x for node in final_path], 
                [node.y for node in final_path], 
                s=10, 
                c=[node.cc for node in final_path], 
                cmap='jet',
                label="path risk value:\nmin: %.6f\nmax: %.6f\navg: %.6f"%(path_min, path_max, path_avg))
    # plt.plot([node.x for node in final_path],
    #         [node.y for node in final_path],
    #         c='k',
    #         label="path risk value:\nmin: %.3f\nmax: %.3f\navg: %.3f"%(path_min, path_max, path_avg))
    plt.colorbar()
    plt.axis([0.0, 14.0, 0.0, 50.0])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.show()

    

if __name__ == '__main__':
    main()
