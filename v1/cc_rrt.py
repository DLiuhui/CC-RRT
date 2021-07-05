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

show_animation = True

class Vehicle:
    def __init__(self):
        self.l_f = 2.7 # 前距离
        self.l_r = 2.5 # 后距离
        self.w = 2.0 # 车宽

class CCRRT:
    """
    Class for CCRRT planning
    """
    class Node:
        def __init__(self, x, y, yaw):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.conv = np.zeros((3,3)) # uncertainty matrix
            self.parent = None
            self.time = 0.0 # travaling time, for calculate cost
            self.cc = 0.0 # chance constraint, for calculate cost
            self.cost = 0.0  # cost = f(time, cc)
            self.cost_lb = 0.0  # cost lower bound
            self.cost_ub = math.inf  # cost upper bound

    def __init__(self, car, start, goal, obstacle_list, rand_area):
        """
        Setting Parameter
        start:Start Position [x,y,yaw]
        goal:Goal Position [x,y,yaw]
        obstacleList:obstacle
        randArea:Random Sampling Area [min,max]
        """
        self.car = car
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.obstacle_list = obstacle_list
        self.get_obs_points_for_collision_checking() # get self.obstacle_list_points
        
        assert len(rand_area) == 4, "rand_area = [x-min, x-max, y-min, y-max]"
        self.min_rand_x = rand_area[0]
        self.max_rand_x = rand_area[1]
        self.min_rand_y = rand_area[2]
        self.max_rand_y = rand_area[3]

        # self.path_resolution = 1.0
        # self.goal_sample_rate = 10 # goal_sample_rate% set goal as sampling node
        
        self.node_list = []
        self.max_iter = 200

        self.max_n_path = 80 # save no more than n_path feasible path to choose
        self.n_path_when_change_strategy = 25
        self.max_n_node = 2500 # save no more than max_node nodes on tree
        self.delta_time = 0.1 # dt second
        self.dis_threshold = 1.0 # distance threshold for steering

        self.max_vehicle_turn_rate = np.pi # vehicle max turning rate
        self.max_angle_diff = np.pi / 2.0
        self.max_vehicle_speed = 18.0  # m/s
        self.min_vehicle_speed = 0.0  # m/s

        self.expect_speed = self.max_vehicle_speed / 2.0 # used for hueristic distance calculation
        self.expect_turn_rate = self.max_vehicle_turn_rate / 4.0 # used for hueristic distance calculation

        # params
        self.k_cc = 100 # for chance_constrain value: node_cost = time + k_cc * node.cc
        # for expect dis: heu_value = k_dis * dis() + (1-k_dis) * node_cost
        # or with k_dis percentage sampling dis() and (1-k_dis) sampling node_cost
        # k_node_cost = 1 - k_dis
        self.k_dis_explore = 0.7 # k_dis value when pay more attention to exploration (more radical)
        self.k_dis_exploit = 0.3 # k_dis value when pay more attention to exploit (safer)
        # the same as k_dis / k_node_cost, but used for find a path relatively close to goal
        self.k_dis_when_no_path = 0.85
        
        # PID control matrix
        self.P = np.diag([1.0, 5.0]) # 2*2
        # self.I = np.diag([0.1, 0.5]) # 2*2
        self.D = np.diag([-2.0, -6.5]) # 2*2
        self.max_steer_step = 30 # max n_step for steering

        self.p_safe = 0.85 # p_safe for chance constraint

        self.sigma_x0 = np.diag([0.2, 0.2, 0.1]) # sigma_x0
        self.sigma_control = np.diag([0.0, 0.0]) # control noise
        self.sigma_pose = np.array([
            [0.02, 0.01, 0.00],
            [0.01, 0.02, 0.00],
            [0.00, 0.00, 0.01]
        ])

        self.nearest_node_step = 8 # get nodes to do tree expanding, used in get_nearest_node_index
        self.n_nearest = 15 # get n nearest nodes, used in get_nearest_node_index
        self.steer_back_step = 8 # used after find a path and try connect to goal after steering

        # init
        # self.sigma_x0[0,0] /= self.path_resolution
        # self.sigma_x0[1,1] /= self.path_resolution
        # self.sigma_control[0,0] /= self.path_resolution
        # self.dis_threshold /= self.path_resolution
        # self.max_vehicle_speed /= self.path_resolution
        # self.min_vehicle_speed /= self.path_resolution

        self.start.conv = self.sigma_x0
        self.start.time = 0.0
        self.start.cc = self.get_chance_constrain(self.start)
        self.start.cost = self.get_cost(self.start.time, self.start.cc)
        self.start.cost_lb = self.get_cost_lb(self.start)
        self.start.cost_ub = math.inf

        self.path_end = [] # save path ending point, if len(path_end) >= n_path, we will stop searching to get the best path
        self.path = [] # save the final path

        self.with_metric = False # planning metric item
        print("Begin CC-RRT")

    def get_obs_points_for_collision_checking(self):
        self.obstacle_list_points = []
        for obs in self.obstacle_list:
            l = obs[7]
            w = obs[8]
            x = obs[0]
            y = obs[1]
            yaw = obs[4]
            p = []
            # 此处点顺序为顺时针
            p.append(self.Node(
                x + l * math.cos(yaw) + w * math.sin(yaw),
                y + l * math.sin(yaw) - w * math.cos(yaw),
                0.0
            ))
            p.append(self.Node(
                x - l * math.cos(yaw) + w * math.sin(yaw),
                y - l * math.sin(yaw) - w * math.cos(yaw),
                0.0
            ))
            p.append(self.Node(
                x - l * math.cos(yaw) - w * math.sin(yaw),
                y - l * math.sin(yaw) + w * math.cos(yaw),
                0.0
            ))
            p.append(self.Node(
                x + l * math.cos(yaw) - w * math.sin(yaw),
                y + l * math.sin(yaw) + w * math.cos(yaw),
                0.0
            ))
            p.append(self.Node(
                x + l * math.cos(yaw) + w * math.sin(yaw),
                y + l * math.sin(yaw) - w * math.cos(yaw),
                0.0
            ))
            self.obstacle_list_points.append(p)

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
            if inter_node.cc < 1.0 - self.p_safe and self.is_feasible(inter_node) and self.safe_steer(inter_node):
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

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path
    
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
        # no path to goal
        # for each feasible node
        for idx in range(0, len(feasible_node_list), self.steer_back_step):
            # try connecting node to goal
            node = feasible_node_list[idx]
            tmp_end_node = self.Node(self.end.x, self.end.y, 0.0)
            tmp_end_node.yaw = np.arctan2(
                    tmp_end_node.y - node.y,
                    tmp_end_node.x - node.x,
                    )
            if not self.angle_check(node, tmp_end_node, self.max_angle_diff):
                continue
            node_to_goal_list = self.steer(node, tmp_end_node)
            if len(node_to_goal_list) and self.calc_distance(node_to_goal_list[-1], self.end) < self.dis_threshold: # get to goal from current node
                for node in node_to_goal_list:
                    self.node_list.append(node) # add to tree
                self.path_end.append(node_to_goal_list[-1]) # save the end node of the path

                # metric
                if self.with_metric and len(self.path_end) == 1:
                    self.time_when_find_first_path = time.clock() - self.timer_start
                    self.node_when_find_first_path = len(self.node_list)

                # update upper-bound cost-to-goal of those nodes
                self.backpropogation(node_to_goal_list[-1])
            if len(self.path_end) > self.max_n_path or len(self.node_list) > self.max_n_node:
                break
    
    def vehicle_constraints(self, x, y, yaw):
        """
        calculate vehicle's edge constraints
        return ([a_1, a_2, a_3, a_4], [b1, b2, b3, b4])
        a_i is the unit outward normals of line constraint 单位外法向量
        b_i = a_i^T * x (x is on the line)
        """
        # nodes in counterclockwise
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

        a1 = [p0[0] - p3[0], p0[1] - p3[1], 0.0] # 单位外法向量
        d = math.sqrt(a1[0]**2 + a1[1]**2)
        a1 = [i / d for i in a1]
        b1 = a1[0] * p0[0] + a1[1] * p0[1]

        a2 = [p1[0] - p0[0], p1[1] - p0[1], 0.0] # 单位外法向量
        d = math.sqrt(a2[0]**2 + a2[1]**2)
        a2 = [i / d for i in a2]
        b2 = a2[0] * p1[0] + a2[1] * p1[1]

        a3 = [p2[0] - p1[0], p2[1] - p1[1], 0.0] # 单位外法向量
        d = math.sqrt(a3[0]**2 + a3[1]**2)
        a3 = [i / d for i in a3]
        b3 = a3[0] * p2[0] + a3[1] * p2[1]

        a4 = [p3[0] - p2[0], p3[1] - p2[1], 0.0] # 单位外法向量
        d = math.sqrt(a4[0]**2 + a4[1]**2)
        a4 = [i / d for i in a4]
        b4 = a4[0] * p3[0] + a4[1] * p3[1]

        return ([a1, a2, a3, a4], [b1, b2, b3, b4])
        
    def get_chance_constrain(self, current):
        A, B = self.vehicle_constraints(current.x, current.y, current.yaw)
        delta_t = 0 # sum(min delte_tj)
        # cal for each obs
        for obs in self.obstacle_list:

            # rotate = np.zeros((3,3))
            # rotate[2,2] = 1.0
            # rotate[0,0] = math.cos(obs[4])
            # rotate[0,1] = -math.sin(obs[4])
            # rotate[1,0] = math.sin(obs[4])
            # rotate[1,1] = math.cos(obs[4])
            angle = abs(obs[4])
            angle = angle if angle <= math.pi / 2.0 else math.pi - angle

            delta_tj = math.inf
            for a, b in zip(A, B):
                a = np.array([a]) # 1*3
                x = np.array([[obs[0]], [obs[1]], [0.0]]) # 3*1
                # abs_mat = np.diag([obs[2], obs[3], obs[4]])
                abs_mat = np.diag([obs[3]*math.sin(angle) + obs[2]*math.cos(angle), obs[2]*math.sin(angle) + obs[3]*math.cos(angle), obs[4]])
                sigma = current.conv + abs_mat
                erf_item = (a.dot(x).item() - b) / np.sqrt(2 * a.dot(sigma).dot(a.transpose()).item())
                cc = 0.5 * (1 - erf(erf_item))
                if cc < delta_tj:
                    delta_tj = cc
            delta_t += delta_tj
        return delta_t
    
    def check_chance_constrain(self, current, p_safe):
        return self.get_chance_constrain(current) < 1 - p_safe
                
    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)
    
    def get_heuristic_dis(self, from_node, to_node):
        """
        heuristic distance function
        """
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun
        t = random.random()
        if len(self.path_end) < self.n_path_when_change_strategy:
            if t < self.k_dis_explore:
                return dis / self.expect_speed + angle / self.expect_turn_rate
            else:
                return from_node.cost
        else:
            if t < self.k_dis_exploit:
                return dis / self.expect_speed + angle / self.expect_turn_rate
            else:
                return from_node.cost
    
    def get_expect_time_to_goal(self, from_node):
        dis, angle = self.calc_distance_and_angle(from_node, self.end)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        return (1 - self.k_dis_when_no_path) * from_node.cost + self.k_dis_when_no_path * (dis / self.expect_speed + angle / self.expect_turn_rate)

    def get_cost_lb(self, node):
        """
        cost lower bound is the expected driving time from current node to the goal
        """
        dis, angle = self.calc_distance_and_angle(node, self.end)
        angle = abs(self.angle_wrap(angle - node.yaw))
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
            node.cost_ub = min(min_child_upper_bound + self.delta_time + node.cc * self.k_cc, node.cost_ub)
            min_child_upper_bound = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            node = node.parent

    def get_nearest_node_index(self, node_list, rnd_node, n_nearest = 1, n_step = 0):
        """
        sort tree node according to heuristic, ascending
        get node from sorted nodes ids, node_ids = [0, n_step, 2*n_step, ...]
        if n_nearest = 1, choose the nearest node (return one node [node_id])
        else return node_ids[:n_nearest]
        """
        # dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
        #          ** 2 for node in node_list]
        dlist = [self.get_heuristic_dis(node, rnd_node) for node in node_list]
        if n_nearest > 1:
            sorted_ids = np.argsort(dlist)
            if not len(sorted_ids) > n_nearest:
                return sorted_ids
            else:
                return sorted_ids[range(0, len(sorted_ids), n_step)][:n_nearest]
        else:
            minind = dlist.index(min(dlist))
            return [minind]
    
    def get_close_to_goal_index(self, node_list):
        """
        get the node close to goal
        """
        dlist = [self.get_expect_time_to_goal(node) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def get_random_node(self):
        """
        node sampling
        """
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
        return rnd
    
    def get_cost(self, time, chance_constraint):
        return time + chance_constraint * self.k_cc
    
    def angle_check(self, node1, node2, max_angle):
        return np.abs(self.angle_wrap(node1.yaw - node2.yaw)) <= max_angle
    
    def isRayIntersectsSegment(self, node, s_poi, e_poi):
        poi = [node.x, node.y]
        #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
        if s_poi[1] == e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
            return False
        if s_poi[1] > poi[1] and e_poi[1] > poi[1]: #线段在射线上边
            return False
        if s_poi[1] < poi[1] and e_poi[1] < poi[1]: #线段在射线下边
            return False
        if s_poi[1] == poi[1] and e_poi[1] > poi[1]: #交点为下端点，对应spoint
            return False
        if e_poi[1] == poi[1] and s_poi[1] > poi[1]: #交点为下端点，对应epoint
            return False
        if s_poi[0] < poi[0] and e_poi[1] < poi[1]: #线段在射线左边
            return False

        xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1]) #求交
        if xseg < poi[0]: #交点在射线起点的左侧
            return False
        return True  #排除上述情况之后

    def is_node_in_vehicle(self, node, nodes):
        p = nodes # 顺时针排列的nodes
        intersection = 0
        for i in range(len(p) - 1):
            s_poi = [p[i].x, p[i].y]
            e_poi = [p[i + 1].x, p[i + 1].y]
            if self.isRayIntersectsSegment(node, s_poi, e_poi):
                intersection += 1
        return True if intersection % 2 == 1 else False

    def collision_checking(self, node):
        """
        碰撞检测
        安全返回 True
        """
        # 自车角点 顺时针
        w = self.car.w / 2.0
        p_ego = []
        p_ego.append(self.Node(
            node.x + self.car.l_f * math.cos(node.yaw) + w * math.sin(node.yaw),
            node.y + self.car.l_f * math.sin(node.yaw) - w * math.cos(node.yaw),
            0.0))
        p_ego.append(self.Node(
            node.x - self.car.l_r * math.cos(node.yaw) + w * math.sin(node.yaw),
            node.y - self.car.l_r * math.sin(node.yaw) - w * math.cos(node.yaw),
            0.0))
        p_ego.append(self.Node(
            node.x - self.car.l_r * math.cos(node.yaw) - w * math.sin(node.yaw),
            node.y - self.car.l_r * math.sin(node.yaw) + w * math.cos(node.yaw),
            0.0))
        p_ego.append(self.Node(
            node.x + self.car.l_f * math.cos(node.yaw) - w * math.sin(node.yaw),
            node.y + self.car.l_f * math.sin(node.yaw) + w * math.cos(node.yaw),
            0.0))
        p_ego.append(self.Node(
            node.x + self.car.l_f * math.cos(node.yaw) + w * math.sin(node.yaw),
            node.y + self.car.l_f * math.sin(node.yaw) - w * math.cos(node.yaw),
            0.0))
        
        l_self = self.calc_distance(p_ego[0], p_ego[2]) / 2.0 # 自车对角线长一半
        for idx, obs in enumerate(self.obstacle_list):
            dis = self.calc_distance(self.Node(obs[0], obs[1], 0.0), node)
            l_obs = math.sqrt(obs[2]**2 + obs[3]**2) # obs对角线长的一半
            if dis > l_self + l_obs:
                continue
            # 检验自车点是否在障碍物内
            for p in p_ego[:4]:
                if self.is_node_in_vehicle(p, self.obstacle_list_points[idx]):
                    return False
            # 检验障碍物点是否在自车内
            for p in self.obstacle_list_points[idx][:4]:
                if self.is_node_in_vehicle(p, p_ego):
                    return False
        return True

    def is_feasible(self, node):
        """
        check if a node is feasible
        """
        # 检查是否在规划区域内
        valid = node.x < self.max_rand_x and node.x > self.min_rand_x and \
                node.y < self.max_rand_y and node.y > self.min_rand_y
        # valid_2 = True
        # # check collision to vehicle bbox
        # for obs in self.obstacle_list:
        #     if self.is_node_in_vehicle(node, obs):
        #         valid_2 = False
        #         break
        # return valid and valid_2
        # 碰撞检测
        return valid and self.collision_checking(node)
        
    def safe_steer(self, node):
        if node.parent:
            xs = np.ones(10) * node.x if node.parent.x == node.x else \
                 np.arange(node.parent.x, node.x, (node.x - node.parent.x) / 10.0)
            ys = np.ones(10) * node.y if node.parent.y == node.y else \
                 np.arange(node.parent.y, node.y, (node.y - node.parent.y) / 10.0)
            yaws = np.ones(10) * node.yaw if node.parent.yaw == node.yaw else \
                   np.arange(node.parent.yaw, node.yaw, (node.yaw - node.parent.yaw) / 10.0)
            for i in range(1, 10):
                # t_node = self.Node(x, y, 0.0)
                # for obs in self.obstacle_list:
                #     if self.is_node_in_vehicle(t_node, obs):
                #         return False
                t_node = self.Node(xs[i], ys[i], yaws[i])
                if not self.is_feasible(t_node):
                    return False
        return True

    def plot_arrow(self, x, y, yaw, length=0.5, width=0.25, fc="r", ec="k"):
        """
        Plot arrow
        """
        if not isinstance(x, float):
            for (ix, iy, iyaw) in zip(x, y, yaw):
                self.plot_arrow(ix, iy, iyaw)
        else:
            plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                    fc=fc, ec=ec, head_width=width, head_length=width)
            plt.plot(x, y)

    def draw_graph(self, rnd=None):
        plt.clf()
        ax = plt.axes()
        # _, ax = plt.subplots()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        for obs in self.obstacle_list:
            ellipse = Ellipse((obs[0], obs[1]), obs[2]*2.0, obs[3]*2.0, math.degrees(obs[4]))
            ax.add_patch(ellipse)

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            plt.plot(node.x, node.y, "*b")
            self.plot_arrow(node.x, node.y, node.yaw, fc='b')

        plt.plot(self.start.x, self.start.y, "xr")
        self.plot_arrow(self.start.x, self.start.y, self.start.yaw, fc='r')
        plt.plot(self.end.x, self.end.y, "xg")
        self.plot_arrow(self.end.x, self.end.y, self.end.yaw, fc='g')
        plt.axis("equal")
        plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
        plt.grid(True)
        plt.pause(0.01)
    
    def draw_path(self):
        for node in self.path:
            plt.plot(node.x, node.y, "*r")
            self.plot_arrow(node.x, node.y, node.yaw, fc='r')
        plt.plot(self.start.x, self.start.y, "*y", markersize=10, label='start')
        self.plot_arrow(self.start.x, self.start.y, self.start.yaw, fc='y')
        plt.plot(self.end.x, self.end.y, "*g", markersize=10, label='goal')
        self.plot_arrow(self.end.x, self.end.y, self.end.yaw, fc='g')
        plt.legend(loc='upper right')
        
    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """
        arg1:from_node -> arg2:to_node
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    @staticmethod
    def calc_distance(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy)
    
    @staticmethod
    def angle_wrap(angle):
        while angle <= -math.pi:
            angle = angle + 2 * math.pi
        while angle > math.pi:
            angle = angle - 2 * math.pi
        return angle

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
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'orange')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'orange')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'orange')

def obstacle_uncertainty_fusion(gts, uncertainties):
    obs = []
    for gt, un in zip(gts, uncertainties):
        a = gt[2] / 2.0
        b = gt[3] / 2.0
        d_a = a * (1 - b * math.sqrt((1 + math.tan(un[2])**2) / (b**2 + a**2 * math.tan(un[2])**2)))
        d_b = d_a / a * b
        # obs.append((gt[0], gt[1], a + d_a + un[0], b + d_b + un[1], gt[4]))
        obs.append((gt[0], gt[1], a + d_a + un[0], b + d_b + un[1], gt[4], d_a + un[0], d_b + un[1], a, b))
    return obs

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
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'k')


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
    cc_rrt = CCRRT(
        car=car,
        start=start,
        goal=goal,
        rand_area=area,
        obstacle_list=obstacle_list)
    # path = cc_rrt.planning(animation=False)
    cc_rrt.planning(animation=False)
    # print(cc_rrt.check_chance_constrain(cc_rrt.end, cc_rrt.p_safe))
    # print(cc_rrt.check_chance_constrain(cc_rrt.start, cc_rrt.p_safe))

    # Draw final path
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

if __name__ == '__main__':
    main()
