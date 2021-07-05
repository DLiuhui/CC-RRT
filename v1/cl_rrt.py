"""
CL-RRT Complete Edition

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

class CLRRT(CCRRT):
    """
    close-loop RRT
    """
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        
        self.max_iter = 300
        self.max_n_path = 80 # save no more than n_path feasible path to choose
        self.n_path_when_change_strategy = 25
        self.max_n_node = 2500 # save no more than max_node nodes on tree

        self.nearest_node_step = 8 # get nodes to do tree expanding, used in get_nearest_node_index
        self.n_nearest = 15 # get n nearest nodes, used in get_nearest_node_index
        self.steer_back_step = 8 # used after find a path and try connect to goal after steering
        print("Begin CL-RRT")
    
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

        return feasible_node_list

    def backpropogation(self, node):
        """
        backpropogation to update cost-upper-bound of a path from start to goal
        the first node is the closest to the goal
        """
        min_child_upper_bound = math.inf # record lowest cost-upper-bound from a node to its childs
        # update upper bound
        # back from goal to root
        while node is not None and self.calc_distance(node, self.end) < self.dis_threshold: # for nodes in the goal region
            node.cost_ub = node.cost_lb
            min_child_upper_bound = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            node = node.parent
        while node is not None: # for nodes out of the goal region
            node.cost_ub = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            min_child_upper_bound = min(min_child_upper_bound + self.delta_time, node.cost_ub)
            node = node.parent
    
    def get_close_to_goal_index(self, node_list):
        """
        get the node close to goal
        """
        dlist = [self.get_expect_time_to_goal(node) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind
    
    def get_cost(self, time, chance_constrain):
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
    cl_rrt = CLRRT(
        car=car,
        start=start,
        goal=goal,
        rand_area=area,
        obstacle_list=obstacle_list)
    # path = cl_rrt.planning(animation=False)
    cl_rrt.planning(animation=False)
    # print(cl_rrt.check_chance_constrain(cl_rrt.end, cl_rrt.p_safe))
    # print(cl_rrt.check_chance_constrain(cl_rrt.start, cl_rrt.p_safe))
    # if path is None:
    #     print("Cannot find path")
    # else:
    #     print("found path!!")

    # # Draw final path
    # if show_animation:
    #     cl_rrt.draw_graph()
    #     plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    #     plt.grid(True)
    #     plt.pause(0.01)  # Need for Mac
    #     plt.show()
    cl_rrt.draw_graph()
    cl_rrt.draw_path()
    draw_vehicle(obstacle_list_gt)
    draw_carsize_of_final_path(car, cl_rrt.path)

    plt.figure(2)
    tmp = [node.cc for node in cl_rrt.path]
    path_min = np.min(tmp)
    path_max = np.max(tmp)
    path_avg = np.average(tmp)
    plt.scatter([node.x for node in cl_rrt.node_list], 
                [node.y for node in cl_rrt.node_list], 
                s=3, 
                c=[node.cc for node in cl_rrt.node_list], 
                cmap='jet')
    plt.plot([node.x for node in cl_rrt.path],
             [node.y for node in cl_rrt.path],
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
