"""
CC-RRT Complete Edition

Algorithm Ref: Chance Constrained RRT fot Probabilistic Robustness to Environment Uncertainty

set all angle as rad
no control noise
"""

import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import erf, erfinv

from v1.cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion, draw_carsize_of_final_path, draw_vehicle

class CCRRT_EXPENDING(CCRRT):

    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_iter = 200
        self.max_n_path = 100 # save no more than n_path feasible path to choose
        self.max_n_node = 3000 # save no more than max_node nodes on tree

        self.nearest_node_step = 1 # get nodes to do tree expanding, used in get_nearest_node_index
        self.n_nearest = 1 # get n nearest nodes, used in get_nearest_node_index
        self.steer_back_step = 8 # used after find a path and try connect to goal after steering

    def planning(self, animation=False, with_metric=False):
        """
        cc_rrt path planning
        animation: flag for animation on or off
        with_metric start metric
        """
        print("Begin CC-RRT")
        self.node_list = [self.start]

        if with_metric:
            self.with_metric = True
            self.node_when_find_first_path = -1
            self.time_when_find_first_path = -1.0
            self.total_time = 0.0
            self.timer_start = time.clock()

        for i in range(self.max_iter):
            if i % 10 == 0:
                print("Iter:", i, ", number of nodes:", len(self.node_list))
            
            sample_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, sample_node, self.n_nearest, self.nearest_node_step)

            for idx in nearest_ind:
                nearest_node = self.node_list[idx]

                # Running some checks in increasing order of computational complexity
                # Do some checking
                # Check 1: Does the sample co-inside with nearest node?
                if self.calc_distance(sample_node, nearest_node) < self.dis_threshold:
                    continue
                sample_node.yaw = np.arctan2(
                    sample_node.y - nearest_node.y,
                    sample_node.x - nearest_node.x,
                    )

                # Check 2: Is the steer angle within the acceptable range?
                if not self.angle_check(nearest_node, sample_node, self.max_angle_diff):
                    continue

                # local planner sampling (also updates sample.conv)
                self.local_planner(nearest_node, sample_node)

                if len(self.path_end) > self.max_n_path or len(self.node_list) > self.max_n_node:
                    break

            # to displaying cc-rrt searching
            # if i % 5 == 0:
            #     self.draw_graph(sample_node)
            #     plt.pause(2.0)
            # self.draw_graph(sample_node)
            # plt.pause(1.0)

            # if len(path_end) >= n_path, we will stop searching to get the best path
            if len(self.path_end) > self.max_n_path or len(self.node_list) > self.max_n_node:
                break

        # end cc_rrt loop    
        print("Tree with %d nodes generated" % len(self.node_list))
    
    def local_planner(self, parent, sample):
        feasible_node_list = self.steer(parent, sample)
        for node in feasible_node_list:
            self.node_list.append(node) # add to tree
        # find a path to goal
        if len(feasible_node_list) and self.calc_distance(feasible_node_list[-1], self.end) < self.dis_threshold:
            self.path_end.append(feasible_node_list[-1]) # save the end node of the path
            
            # metric
            if self.with_metric and len(self.path_end) == 1:
                self.time_when_find_first_path = time.clock() - self.timer_start
                self.node_when_find_first_path = len(self.node_list)

            # back propogation
            self.backpropogation(feasible_node_list[-1])


def main():
    print("Start " + __file__)

    area = [-2, 20, -2, 20] # x-min x-max y-min y-max

    # Set Initial parameters
    start = [7.5, 0.0, np.deg2rad(90.0)]
    goal = [7.5, 18.0, np.deg2rad(90.0)]
    car = Vehicle()

    # ====Search Path with CCRRT====
    # (x, y, vehicle_length, vehicle_width, radius [-pi, pi])
    # axis = length + sigma
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

    # (x, y, long_axis, short_axis, radius [-pi, pi])
    # vehicle_length = long_axis * 2
    # vehicle_width = short_axis * 2
    obstacle_list = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)

    # Set Initial parameters
    cc_rrt = CCRRT_EXPENDING(
        car=car,
        start=start,
        goal=goal,
        rand_area=area,
        obstacle_list=obstacle_list)
    # path = cc_rrt.planning(animation=False)
    cc_rrt.planning(animation=False)
    # print(cc_rrt.check_chance_constrain(cc_rrt.end, cc_rrt.p_safe))
    # print(cc_rrt.check_chance_constrain(cc_rrt.start, cc_rrt.p_safe))
    # if path is None:
    #     print("Cannot find path")
    # else:
    #     print("found path!!")

    # # Draw final path
    # if show_animation:
    #     cc_rrt.draw_graph()
    #     plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    #     plt.grid(True)
    #     plt.pause(0.01)  # Need for Mac
    #     plt.show()
    cc_rrt.draw_graph()
    draw_vehicle(obstacle_list_gt)
    draw_carsize_of_final_path(car, cc_rrt.path)

    plt.figure(2)
    plt.scatter([node.x for node in cc_rrt.node_list], 
                [node.y for node in cc_rrt.node_list], 
                s=3, 
                c=[node.cc for node in cc_rrt.node_list], 
                cmap='jet')
    plt.plot([node.x for node in cc_rrt.path],
             [node.y for node in cc_rrt.path],
             c='k')
    plt.colorbar()
    plt.axis([area[0], area[1], area[2], area[3]])
    plt.grid(True)
    plt.show()
    # plt.savefig("cc-rrt-h-fun-3.png")

if __name__ == '__main__':
    main()
