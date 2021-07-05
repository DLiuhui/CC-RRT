"""
metric for planning algo
metric for different heuristic function
"""

import math
import os
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from cc_rrt import CCRRT, obstacle_uncertainty_fusion, Vehicle

SAVE_ROOT = os.path.abspath(os.path.curdir)

class CCRRT1(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
    
    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 1 balance risk and explore
        return from_node.cost + dis / self.expect_speed + angle / self.expect_turn_rate

class CCRRT2(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
    
    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 2
        return min(from_node.cost, dis / self.expect_speed + angle / self.expect_turn_rate)

class CCRRT3(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
    
    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 3
        return max(from_node.cost, dis / self.expect_speed + angle / self.expect_turn_rate)

class CCRRT4(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
        self.k_dis_explore = 0.7
        self.k_dis_exploit = 0.3
    
    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 4
        dis_to_goal = self.calc_distance(to_node, self.end)
        if dis_to_goal > (self.max_rand_y - self.min_rand_y) / 2.0:
            return (1 - self.k_dis_explore) * from_node.cost + self.k_dis_explore * (dis / self.expect_speed + angle / self.expect_turn_rate)
        else:
            return (1 - self.k_dis_exploit) * from_node.cost + self.k_dis_exploit * (dis / self.expect_speed + angle / self.expect_turn_rate)

class CCRRT5(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
        self.k_dis_explore = 0.7
        self.k_dis_exploit = 0.3

    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 5
        dis_to_goal = self.calc_distance(to_node, self.end)
        t = random.random()
        if dis_to_goal > (self.max_rand_y - self.min_rand_y) / 2.0:
            if t < self.k_dis_explore:
                return dis / self.expect_speed + angle / self.expect_turn_rate
            else:
                return from_node.cost
        else:
            if t < self.k_dis_exploit:
                return dis / self.expect_speed + angle / self.expect_turn_rate
            else:
                return from_node.cost

class CCRRT6(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
        self.k_dis_explore = 0.7
        self.k_dis_exploit = 0.3
        self.n_path_when_change_strategy = 25
    
    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 6
        if len(self.path_end) < self.n_path_when_change_strategy:
            return (1 - self.k_dis_explore) * from_node.cost + self.k_dis_explore * (dis / self.expect_speed + angle / self.expect_turn_rate)
        else:
            return (1 - self.k_dis_exploit) * from_node.cost + self.k_dis_exploit * (dis / self.expect_speed + angle / self.expect_turn_rate)

class CCRRT7(CCRRT):
    def __init__(self, car, start, goal, obstacle_list, rand_area):
        super().__init__(car, start, goal, obstacle_list, rand_area)
        self.max_n_path = 50
        self.max_n_node = 5000
        self.k_dis_explore = 0.7
        self.k_dis_exploit = 0.3
        self.n_path_when_change_strategy = 25
    
    def get_heuristic_dis(self, from_node, to_node):
        dis, angle = self.calc_distance_and_angle(from_node, to_node)
        angle = abs(self.angle_wrap(angle - from_node.yaw))
        # heu fun 7
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

    # 4 obs [1,3,5,7]
    # 6 obs [1,3,4,5,7,8]
    # 8 obs all
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

    save_path = os.path.join(SAVE_ROOT, "planning_metric_%d_obs" % len(obstacle_list_gt))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # (x, y, long_axis, short_axis, radius [-pi, pi])
    # vehicle_length = long_axis * 2
    # vehicle_width = short_axis * 2
    obstacle_list = obstacle_uncertainty_fusion(obstacle_list_gt, obstacle_list_uncertainty)

    experiment_time = 100
    # all cc-rrt iter time set to 100
    epoch = []
    # use to save metric history
    time_when_find_first_path = []
    node_when_find_first_path = []
    n_node = []
    n_path = []
    node_per_path = []
    total_time = []
    total_dis = []
    total_angle = []
    max_cc = []
    avg_cc = []
    min_cc = []
    path_length = []

    cc_rrt_models = [
        CCRRT1,
        CCRRT2,
        CCRRT3,
        CCRRT4,
        CCRRT5,
        CCRRT6,
        CCRRT7
    ]

    for cc_rrt_model in cc_rrt_models:

        epoch_ = []
        time_when_find_first_path_ = []
        node_when_find_first_path_ = []
        n_node_ = []
        n_path_ = []
        node_per_path_ = []
        total_time_ = []
        total_dis_ = []
        total_angle_ = []
        max_cc_ = []
        avg_cc_ = []
        min_cc_ = []
        path_length_ = []
        
        for i in range(experiment_time):
            cc_rrt = cc_rrt_model(car=car, start=start, goal=goal, rand_area=area, obstacle_list=obstacle_list)
            cc_rrt.planning(with_metric=True)
            
            if len(cc_rrt.path_end): # success find a path
                epoch_.append(i + 1)
                # get metric
                time_when_find_first_path_.append(cc_rrt.time_when_find_first_path)
                node_when_find_first_path_.append(cc_rrt.node_when_find_first_path)
                n_node_.append(len(cc_rrt.node_list))
                n_path_.append(len(cc_rrt.path_end))
                node_per_path_.append(float(len(cc_rrt.node_list)) / float(len(cc_rrt.path_end)))
                total_time_.append(cc_rrt.total_time)
            
                pre_node = cc_rrt.path[0]
                t_max_cc = pre_node.cc
                t_min_cc = pre_node.cc
                t_avg_cc = pre_node.cc
                t_total_dis = 0.0
                t_total_angle = 0.0

                for node in cc_rrt.path[1:]:
                    t_total_dis += cc_rrt.calc_distance(pre_node, node)
                    t_total_angle += cc_rrt.angle_wrap(abs(node.yaw - pre_node.yaw))
                    if node.cc > t_max_cc:
                        t_max_cc = node.cc
                    if node.cc < t_min_cc:
                        t_min_cc = node.cc
                    t_avg_cc += node.cc
                    pre_node = node

                total_dis_.append(t_total_dis)
                total_angle_.append(t_total_angle)
                max_cc_.append(t_max_cc)
                min_cc_.append(t_min_cc)
                avg_cc_.append(t_avg_cc / float(len(cc_rrt.path)))
                path_length_.append(len(cc_rrt.path))
            del cc_rrt
        
        epoch.append(epoch_)
        # use to save metric history
        time_when_find_first_path.append(time_when_find_first_path_)
        node_when_find_first_path.append(node_when_find_first_path_)
        n_node.append(n_node_)
        n_path.append(n_path_)
        node_per_path.append(node_per_path_)
        total_time.append(total_time_)
        total_dis.append(total_dis_)
        total_angle.append(total_angle_)
        max_cc.append(max_cc_)
        min_cc.append(min_cc_)
        avg_cc.append(avg_cc_)
        path_length.append(path_length_)

    n_algo = len(cc_rrt_models)

    f = open(save_path + "metric.txt", 'w')
    f.write("p_safe = %.2f, with %d obs\n" % (0.80, len(obstacle_list_gt)))
    f.write("\n")
    
    f.write("success rate\n")
    for i in range(n_algo):
        f.write("heu-%d "%(i+1))
        f.write("%.2f, %d / %d\n"%(len(epoch[i]) / float(experiment_time), len(epoch[i]), experiment_time))
    f.write("\n")
    
    f.write("找到第一条路径的时间\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
            np.mean(time_when_find_first_path[i]),
            np.std(time_when_find_first_path[i]),
            np.min(time_when_find_first_path[i]),
            np.max(time_when_find_first_path[i])
        ))
    f.write("\n")

    f.write("找到第一条路径的节点数\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
            np.mean(node_when_find_first_path[i]),
            np.std(node_when_find_first_path[i]),
            np.min(node_when_find_first_path[i]),
            np.max(node_when_find_first_path[i])
        ))
    f.write("\n")
    
    f.write("最优路径长\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
            np.mean(total_dis[i]),
            np.std(total_dis[i]),
            np.min(total_dis[i]),
            np.max(total_dis[i])
        ))
    f.write("\n")
    
    f.write("最优路径转角累计值\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
            np.mean(total_angle[i]),
            np.std(total_angle[i]),
            np.min(total_angle[i]),
            np.max(total_angle[i])
        ))
    f.write("\n")

    f.write("最优路径节点数\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
            np.mean(path_length[i]),
            np.std(path_length[i]),
            np.min(path_length[i]),
            np.max(path_length[i])
        ))
    f.write("\n")

    f.write("最优路径最大cc(风险)值\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.6f%-8.6f%-8.6f%-8.6f\n"%(
            np.mean(max_cc[i]),
            np.std(max_cc[i]),
            np.min(max_cc[i]),
            np.max(max_cc[i])
        ))
    f.write("\n")

    f.write("最优路径最小cc(风险)值\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.6f%-8.6f%-8.6f%-8.6f\n"%(
            np.mean(min_cc[i]),
            np.std(min_cc[i]),
            np.min(min_cc[i]),
            np.max(min_cc[i])
        ))
    f.write("\n")

    f.write("最优路径平均cc(风险)值\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.6f%-8.6f%-8.6f%-8.6f\n"%(
            np.mean(avg_cc[i]),
            np.std(avg_cc[i]),
            np.min(avg_cc[i]),
            np.max(avg_cc[i])
        ))
    f.write("\n")

    # f.write("规划过程找到的节点总数\n")
    # f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    # for i in range(n_algo):
    #     f.write("heu-%d  "%(i+1))
    #     f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
    #         np.mean(n_node[i]),
    #         np.std(n_node[i]),
    #         np.min(n_node[i]),
    #         np.max(n_node[i])
    #     ))
    # f.write("\n")

    # f.write("规划过程找到的路径总数\n")
    # f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    # for i in range(n_algo):
    #     f.write("heu-%d  "%(i+1))
    #     f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
    #         np.mean(n_path[i]),
    #         np.std(n_path[i]),
    #         np.min(n_path[i]),
    #         np.max(n_path[i])
    #     ))
    # f.write("\n")

    # f.write("规划过程 总节点/总路径数\n")
    # f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    # for i in range(n_algo):
    #     f.write("heu-%d  "%(i+1))
    #     f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
    #         np.mean(node_per_path[i]),
    #         np.std(node_per_path[i]),
    #         np.min(node_per_path[i]),
    #         np.max(node_per_path[i])
    #     ))
    # f.write("\n")

    f.write("规划过程总时间\n")
    f.write(" "*7 + "avg" + " "*5 + "std" + " "*5 + "min" + " "*5 + "max" + " "*5 + "\n")
    for i in range(n_algo):
        f.write("heu-%d  "%(i+1))
        f.write("%-8.2f%-8.2f%-8.2f%-8.2f\n"%(
            np.mean(total_time[i]),
            np.std(total_time[i]),
            np.min(total_time[i]),
            np.max(total_time[i])
        ))
    f.write("\n")

    f.close()

    # draw
    step = max(experiment_time / 10, 1)
    colors = ['k', 'r', 'g', 'c', 'b', 'm', 'y']
    x_ticks = np.arange(0, experiment_time + step, step)
    x_ticks[0] = 1
    x_ticks[-1] = experiment_time
    
    # time_when_find_first_path
    plt.figure(1, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(time_when_find_first_path[i])
        plt.plot(epoch[i], time_when_find_first_path[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Time cost when find first path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "time_when_find_first_path.png"))

    # node_when_find_first_path
    plt.figure(2, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(node_when_find_first_path[i])
        plt.plot(epoch[i], node_when_find_first_path[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Node number when find first path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "node_when_find_first_path.png"))

    # n_node
    # plt.figure(3, figsize=(9, 6))
    # for i in range(n_algo):
    #     avg = np.mean(n_node[i])
    #     plt.plot(epoch[i], n_node[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
    #     plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    # plt.xlabel('Experiment Time')
    # plt.ylabel('Total node number in expand tree')
    # plt.xticks(x_ticks)
    # plt.grid(True)
    # plt.legend(loc='upper right')
    # plt.savefig(save_path + "n_node.png")

    # n_path
    # plt.figure(4, figsize=(9, 6))
    # for i in range(n_algo):
    #     avg = np.mean(n_path[i])
    #     plt.plot(epoch[i], n_path[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
    #     plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    # plt.xlabel('Experiment Time')
    # plt.ylabel('Total found path')
    # plt.xticks(x_ticks)
    # plt.grid(True)
    # plt.legend(loc='upper right')
    # plt.savefig(save_path + "n_path.png")

    # node_per_path = []
    # plt.figure(5, figsize=(9, 6))
    # for i in range(n_algo):
    #     avg = np.mean(node_per_path[i])
    #     plt.plot(epoch[i], node_per_path[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
    #     plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    # plt.xlabel('Experiment Time')
    # plt.ylabel('Node per found a path')
    # plt.xticks(x_ticks)
    # plt.grid(True)
    # plt.legend(loc='upper right')
    # plt.savefig(save_path + "node_per_path.png")

    # total_time = []
    plt.figure(6, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(total_time[i])
        plt.plot(epoch[i], total_time[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Total planning time')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "total_time.png"))

    # total_dis = []
    plt.figure(7, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(total_dis[i])
        plt.plot(epoch[i], total_dis[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Length of the best path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "total_dis.png"))

    # total_angle = []
    plt.figure(8, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(total_angle[i])
        plt.plot(epoch[i], total_angle[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Turning rate of the best path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "total_angle.png"))

    # max_cc = []
    plt.figure(9, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(max_cc[i])
        plt.plot(epoch[i], max_cc[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Max possible to collision of the best path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "max_cc.png"))

    # min_cc = []
    plt.figure(10, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(min_cc[i])
        plt.plot(epoch[i], min_cc[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Min possible to collision of the best path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "min_cc.png"))

    # avg_cc = []
    plt.figure(11, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(avg_cc[i])
        plt.plot(epoch[i], avg_cc[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Average possible to collision of the best path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "avg_cc.png"))

    # path_length = []
    plt.figure(12, figsize=(9, 6))
    for i in range(n_algo):
        avg = np.mean(path_length[i])
        plt.plot(epoch[i], path_length[i], color=colors[i], label="heu-%d avg %.4f"%(i+1, avg))
        plt.plot(epoch[i], [avg] * len(epoch[i]), color=colors[i], linestyle='--')
    plt.xlabel('Experiment Time')
    plt.ylabel('Node number of the best path')
    plt.xticks(x_ticks)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "path_length.png"))

if __name__ == '__main__':
    main()
