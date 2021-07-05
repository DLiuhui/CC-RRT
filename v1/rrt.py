"""
RRT Complete Edition

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
from cc_rrt import CCRRT, Vehicle, obstacle_uncertainty_fusion, draw_vehicle, draw_carsize_of_final_path

class RRT(CCRRT):
    """
    basic RRT
    """
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)

        # self.path_resolution = 1.0
        self.goal_sample_rate = 10 # goal_sample_rate% set goal as sampling node
        self.node_list = []
        self.max_iter = 2000

        self.max_n_path = 100 # save no more than n_path feasible path to choose
        self.max_n_node = 3000 # save no more than max_node nodes on tree

        self.nearest_node_step = 1 # get nodes to do tree expanding, used in get_nearest_node_index
        self.n_nearest = 1 # get n nearest nodes, used in get_nearest_node_index
        self.steer_back_step = 8 # used after find a path and try connect to goal after steering
        print("Begin RRT")

    def planning(self, animation=False, with_metric=False):
        """
        cc_rrt path planning
        animation: flag for animation on or off
        with_metric start metric
        """
        self.node_list = [self.start]

        if with_metric:
            self.with_metric = True
            self.node_when_find_first_path = -1
            self.time_when_find_first_path = -1.0
            self.total_time = 0.0
            self.timer_start = time.clock()

        for i in range(self.max_iter):
            if i % 100 == 0:
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

        if len(self.path_end):
            print("%d path be found" % len(self.path_end))
            # choosing best path
            min_upper_bound = math.inf
            final_goal_node_idx = -1
            for idx, node in enumerate(self.path_end):
                # max
                path_max_upper_bound = -math.inf
                while node:
                    path_max_upper_bound = max(path_max_upper_bound, node.cost_ub)
                    node = node.parent
                if path_max_upper_bound < min_upper_bound:
                    min_upper_bound = path_max_upper_bound
                    final_goal_node_idx = idx
                # sum
                # path_sum_upper_bound = 0.0
                # while node:
                #     path_sum_upper_bound += node.cost_ub
                #     node = node.parent
                # if path_sum_upper_bound < min_upper_bound:
                #     min_upper_bound = path_sum_upper_bound
                #     final_goal_node_idx = idx
            final_goal_node = self.path_end[final_goal_node_idx]
        else:
            print("No path found!")
            # choose a feasible path can drive to goal closer
            nearest_ind = self.get_close_to_goal_index(self.node_list)
            final_goal_node = self.node_list[nearest_ind]
        
        # return path
        while final_goal_node:
            self.path.append(final_goal_node)
            final_goal_node = final_goal_node.parent
        
        if self.with_metric:
            self.total_time = time.clock() - self.timer_start

        return self.path
    
    def steer(self, from_node, to_node):
        """
        steer with chance constrain checking
        begin: from_node
        return path = [inter_node, ..., inter_node, to_node(if feasible)]
        """
        # reference v & w
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = self.angle_wrap(angle - from_node.yaw)

        if abs(angle) > math.pi / 3.0:
            self.P[0,0] = 0.05
            self.D[0,0] = -0.10
            self.min_vehicle_speed = 1.0
            self.max_vehicle_turn_rate = math.pi
        elif abs(angle) > math.pi / 6.0:
            self.P[0,0] = 0.25
            self.D[0,0] = -0.5
            self.min_vehicle_speed = 6.5
            self.max_vehicle_turn_rate = math.pi
        else:
            self.P[0,0] = 1.0
            self.D[0,0] = -2.0
            self.min_vehicle_speed = 13.0
            self.max_vehicle_turn_rate = math.pi / 2.0

        u_p = self.P.dot(np.array([[dis],[angle]]))
        u_d = np.zeros((2,1))
        u = u_p + u_d
        u[0,0] = max(self.min_vehicle_speed, min(u[0,0], self.max_vehicle_speed))
        if abs(u[1,0]) > self.max_vehicle_turn_rate:
            u[1,0] = np.sign(angle) * self.max_vehicle_turn_rate
        
        # prev node: deep copy from the from_node
        prev = self.Node(from_node.x, from_node.y, from_node.yaw) # local init
        prev.conv = from_node.conv
        prev.cost = from_node.cost
        prev.parent = from_node.parent
        prev.time = from_node.time
        prev.cc = from_node.cc
        prev.cost_lb = from_node.cost_lb
        prev.cost_ub = from_node.cost_ub

        prev_dis = dis
        prev_angle = angle

        J1 = np.diag([1.0, 1.0, 1.0])

        J2 = np.zeros((3,2))
        J2[0,0] = self.delta_time * math.cos(prev.yaw)
        J2[1,0] = self.delta_time * math.sin(prev.yaw)
        J2[2,1] = self.delta_time

        # get feasible node from N_near->N_sample
        feasible_node_list = []
        n_step = 0
        # feaisble_to_end = True
        while self.calc_distance(prev, to_node) > self.dis_threshold and n_step < self.max_steer_step:
            pose = J1.dot(np.array([[prev.x], [prev.y], [prev.yaw]])) + J2.dot(u)
            inter_node = self.Node(pose[0].item(), pose[1].item(), pose[2].item())
            inter_node.yaw = self.angle_wrap(inter_node.yaw)
            inter_node.parent = prev
            inter_node.conv = J1.dot(prev.conv).dot(J1.transpose()) + \
                              J2.dot(self.sigma_control).dot(J2.transpose()) + \
                              self.sigma_pose
            inter_node.cc = self.get_chance_constrain(inter_node)
            if self.is_feasible(inter_node) and self.safe_steer(inter_node):
                inter_node.time = prev.time + self.delta_time
                inter_node.cost = self.get_cost(inter_node.time, inter_node.cc)
                inter_node.cost_lb = self.get_cost_lb(inter_node)
                feasible_node_list.append(inter_node)
                prev = inter_node # inter_node will point to the next inter_node
                
                # update J1 J2
                J2[0,0] = self.delta_time * math.cos(prev.yaw)
                J2[1,0] = self.delta_time * math.sin(prev.yaw)
                J2[2,1] = self.delta_time

                dis, angle = self.calc_distance_and_angle(prev, to_node)
                angle = self.angle_wrap(angle - prev.yaw)

                if abs(angle) > math.pi / 3.0:
                    self.P[0,0] = 0.05
                    self.D[0,0] = -0.10
                    self.min_vehicle_speed = 1.0
                    self.max_vehicle_turn_rate = math.pi
                elif abs(angle) > math.pi / 6.0:
                    self.P[0,0] = 0.25
                    self.D[0,0] = -0.5
                    self.min_vehicle_speed = 6.5
                    self.max_vehicle_turn_rate = math.pi
                else:
                    self.P[0,0] = 1.0
                    self.D[0,0] = -2.0
                    self.min_vehicle_speed = 13.0
                    self.max_vehicle_turn_rate = math.pi / 2.0

                u_p = self.P.dot(np.array([[dis],[angle]]))
                u_d = self.D.dot(np.array([[dis - prev_dis], [angle - prev_angle]]))
                u = u_p + u_d
                u[0,0] = max(self.min_vehicle_speed, min(u[0,0], self.max_vehicle_speed))
                if abs(u[1,0]) > self.max_vehicle_turn_rate:
                    u[1,0] = np.sign(angle) * self.max_vehicle_turn_rate
                prev_dis = dis
                prev_angle = angle
                n_step += 1
            else:
                # feaisble_to_end = False
                break
        
        # add to_node if possible
        # if feaisble_to_end and n_step < self.max_steer_step:
        #     to_node.parent = prev
        #     to_node.conv = J1.dot(prev.conv).dot(J1.transpose()) + \
        #                    J2.dot(self.sigma_control).dot(J2.transpose()) + \
        #                    self.sigma_pose
        #     to_node.cc = self.get_chance_constrain(to_node)
        #     if to_node.cc < 1 - self.p_safe:
        #         to_node.time = prev.time + self.delta_time
        #         to_node.cost = self.get_cost(to_node.time, to_node.cc)
        #         to_node.cost_lb = self.get_cost_lb(to_node)
        #         feasible_node_list.append(to_node)

        return feasible_node_list
    
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
            return
    
    def get_heuristic_dis(self, from_node, to_node):
        """
        heuristic distance function
        """
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun
        return dis / self.expect_speed + angle / self.expect_turn_rate
    
    def backpropogation(self, node):
        """
        backpropogation to update cost-upper-bound of a path from start to goal
        the first node is the closest to the goal
        """
        min_child_upper_bound = math.inf # record lowest cost-upper-bound from a node to its childs
        # update upper bound
        
        # back from root to goal
        # while node is not None and self.calc_distance(node, self.end) < self.dis_threshold: # for nodes in the goal region
        #     node.cost_ub = node.cost_lb
        #     min_child_upper_bound = min(min_child_upper_bound, node.cost + node.cost_ub)
        #     node = node.parent
        # while node is not None: # for nodes out of the goal region
        #     node.cost_ub = min(min_child_upper_bound, node.cost_ub)
        #     min_child_upper_bound = min(min_child_upper_bound, node.cost + node.cost_ub)
        #     node = node.parent

        # back from goal to root
        while node is not None and self.calc_distance(node, self.end) < self.dis_threshold: # for nodes in the goal region
            node.cost_ub = node.cost_lb
            min_child_upper_bound = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            node = node.parent
        while node is not None: # for nodes out of the goal region
            node.cost_ub = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            min_child_upper_bound = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            node = node.parent

    def get_random_node(self):
        """
        node sampling
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            while True:
                rnd = self.Node(random.uniform(self.min_rand_x, self.max_rand_x),
                                random.uniform(self.min_rand_y, self.max_rand_y),
                                0.0)
                valid = True
                # discard point in obstacle range
                for obs in self.obstacle_list_points:
                    if self.is_node_in_vehicle(rnd, obs):
                        valid = False
                        break
                if valid:
                    break
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, 0.0)
        return rnd
    
    def get_cost(self, time, chance_constraint):
        return time


def main():
    print("Start " + __file__)

    area = [-2, 20, -2, 20] # x-min x-max y-min y-max

    # Set Initial parameters
    start = [7.5, -1.0, np.deg2rad(90.0)]
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
    cc_rrt = RRT(
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
    cc_rrt.draw_path()
    draw_vehicle(obstacle_list_gt)
    draw_carsize_of_final_path(car, cc_rrt.path)

    plt.figure(2)
    tmp = [node.cc for node in cc_rrt.path]
    path_min = np.min(tmp)
    path_max = np.max(tmp)
    path_avg = np.average(tmp)
    plt.scatter([node.x for node in cc_rrt.node_list], 
                [node.y for node in cc_rrt.node_list], 
                s=3, 
                c=[node.cc for node in cc_rrt.node_list], 
                cmap='jet')
    plt.plot([node.x for node in cc_rrt.path],
             [node.y for node in cc_rrt.path],
             c='k',
             label="path risk value:\nmin: %.6f\nmax: %.6f\navg: %.6f"%(path_min, path_max, path_avg))
    plt.colorbar()
    plt.axis([area[0], area[1], area[2], area[3]])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    # plt.savefig("cc-rrt-h-fun-3.png")

if __name__ == '__main__':
    main()
