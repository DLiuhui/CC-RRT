"""
ackerman vehicle model
simple pid steering model

set all angle as rad
"""

import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

def plot_arrow(x, y, yaw, length=0.5, width=0.25, fc="r", ec="k"):
    """
    Plot arrow
    """
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def draw_graph(area, start, end, inter):
    plt.clf()
    # _, ax = plt.subplots()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
                                    lambda event: [exit(0) if event.key == 'escape' else None])
    for node in inter:
        # if node.parent:
        #     plt.plot(node.path_x, node.path_y, "-b")
        plt.plot(node.x, node.y, "*b")
        plot_arrow(node.x, node.y, node.yaw, fc='b')

    plt.plot(start.x, start.y, "xr")
    plot_arrow(start.x, start.y, start.yaw, fc='r')
    plt.plot(end.x, end.y, "xg")
    plot_arrow(end.x, end.y, end.yaw, fc='g')
    plt.axis("equal")
    # plt.axis([area[0], area[1], area[0], area[1]])
    plt.grid(True)
    plt.pause(0.01)


def calc_distance_and_angle(from_node, to_node):
    """
    arg1:from_node -> arg2:to_node
    """
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    d = math.hypot(dx, dy)
    # theta = to_node.yaw - from_node.yaw
    theta = math.atan2(dy, dx)
    return d, theta

def distance(from_node, to_node):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    return math.hypot(dx, dy)

def angle_wrap(angle):
    while angle <= -math.pi:
        angle = angle + 2 * math.pi
    while angle > math.pi:
        angle = angle - 2 * math.pi
    return angle

def get_random_node():
    """
    node sampling
    """
    rnd = Node(random.uniform(0, 50),
               random.uniform(0, 50),
               np.deg2rad(random.uniform(-180, 180)))
    return rnd

def main():
    area = [0, 50]
    # Set Initial parameters
    # start = Node(5.0, 0.0, np.deg2rad(60.0))
    # end = Node(1.0, 12.0, np.deg2rad(90.0))
    start = get_random_node()
    end = get_random_node()
    
    max_vehicle_speed = 18.0
    # min_vehicle_speed = 8.5
    min_vehicle_speed = 0.0

    max_vehicle_turn_rate = math.pi

    delta_time = 0.1

    delta_dis = 1.0
    max_steps = 100

    P = np.diag([1.0, 5.0]) # 2*2
    # I = np.diag([0.1, 0.5]) # 2*2
    D = np.diag([-2.0, -6.5]) # 2*2

    dis, angle = calc_distance_and_angle(start, end)
    angle = angle_wrap(angle - start.yaw)
    
    if abs(angle) > math.pi / 3.0:
        P[0,0] = 0.05
        D[0,0] = -0.10
        min_vehicle_speed = 1.0
        max_vehicle_turn_rate = math.pi
    elif abs(angle) > math.pi / 6.0:
        P[0,0] = 0.25
        D[0,0] = -0.5
        min_vehicle_speed = 6.5
        max_vehicle_turn_rate = math.pi
    else:
        P[0,0] = 1.0
        D[0,0] = -2.0
        min_vehicle_speed = 13.0
        max_vehicle_turn_rate = math.pi / 2.0

    u_p = P.dot(np.array([[dis],[angle]]))
    # u_i = I.dot(np.array([[dis],[angle]]))
    u_d = np.zeros((2,1))
    # u = u_p + u_i + u_d
    u = u_p + u_d
    u[0,0] = max(min_vehicle_speed, min(u[0,0], max_vehicle_speed))
    if abs(u[1,0]) > max_vehicle_turn_rate:
        u[1,0] = np.sign(angle) * max_vehicle_turn_rate

    prev = Node(start.x, start.y, start.yaw)
    prev_dis = dis
    prev_angle = angle
    # total_dis = dis
    # total_angle = angle

    J1 = np.diag([1.0, 1.0, 1.0])

    J2 = np.zeros((3,2))
    J2[0,0] = delta_time * math.cos(prev.yaw)
    J2[1,0] = delta_time * math.sin(prev.yaw)
    J2[2,1] = delta_time

    inter = []
    u_his = [u[0,0]]
    w_his = [u[1,0]]
    n_steps = 0
    while distance(prev, end) > delta_dis and n_steps < max_steps:
        pose = J1.dot(np.array([[prev.x],[prev.y],[prev.yaw]])) + J2.dot(u)
        new_pose = Node(pose[0].item(), pose[1].item(), pose[2].item())

        inter.append(new_pose)
        # draw
        # draw_graph(area, start, end, inter)
        # plt.pause(0.3)
        # update
        prev = new_pose

        J2[0,0] = delta_time * math.cos(prev.yaw)
        J2[1,0] = delta_time * math.sin(prev.yaw)
        J2[2,1] = delta_time

        dis, angle = calc_distance_and_angle(prev, end)
        # total_dis += dis
        # total_angle += angle
        angle = angle_wrap(angle - prev.yaw)
        
        if abs(angle) > math.pi / 3.0:
            P[0,0] = 0.05
            D[0,0] = -0.10
            min_vehicle_speed = 1.0
            max_vehicle_turn_rate = math.pi
        elif abs(angle) > math.pi / 6.0:
            P[0,0] = 0.25
            D[0,0] = -0.5
            min_vehicle_speed = 6.5
            max_vehicle_turn_rate = math.pi
        else:
            P[0,0] = 1.0
            D[0,0] = -2.0
            min_vehicle_speed = 13.0
            max_vehicle_turn_rate = math.pi / 2.0

        u_p = P.dot(np.array([[dis],[angle]]))
        # u_i = I.dot(np.array([[total_dis],[total_angle]]))
        u_d = D.dot(np.array([[dis - prev_dis],[angle - prev_angle]]))
        # u = u_p + u_i + u_d
        u = u_p + u_d
        u[0,0] = max(min_vehicle_speed, min(u[0,0], max_vehicle_speed))
        if abs(u[1,0]) > max_vehicle_turn_rate:
            u[1,0] = np.sign(angle) * max_vehicle_turn_rate
        prev_dis = dis
        prev_angle = angle

        n_steps += 1

        u_his.append(u[0,0])
        w_his.append(u[1,0])

    print(n_steps)
    draw_graph(area, start, end, inter)
    plt.figure(2)
    plt.plot(list(range(n_steps + 1)), u_his)
    plt.figure(3)
    plt.plot(list(range(n_steps + 1)), w_his)
    plt.show()

if __name__ == '__main__':
    main()
