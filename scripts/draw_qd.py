#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: draw_qd.py
Date: 2023/3/23 上午9:31
Description: 
"""
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, Vector3
import numpy as np


def draw_mul_qd(num_agent: int) -> MarkerArray:
    marker_array = MarkerArray()
    for i in range(num_agent):
        marker = draw_one_qd(i)
        marker_array.markers.append(marker)
    return marker_array


def draw_one_qd(idx: int) -> Marker:
    pi = np.pi

    cm = 0.01  # unit: cm
    prop_r = 8 * cm  # radius of 6inch propeller
    wheelbase_p = 20 * cm  # 0.2744 * sqrt(2) / 2
    h = 5 * cm  # height of the quadrotor

    motor_1 = [wheelbase_p / 2, -wheelbase_p / 2]
    motor_2 = [wheelbase_p / 2, wheelbase_p / 2]
    motor_3 = [-wheelbase_p / 2, wheelbase_p / 2]
    motor_4 = [-wheelbase_p / 2, -wheelbase_p / 2]

    triangle_l = 8 * cm
    triangle_w = 4 * cm

    # points are in FLU coordinates
    p0 = Point(triangle_l / 2, 0, 0)  # triangle on the body
    p1 = Point(-triangle_l / 2, triangle_w / 2, 0)
    p2 = Point(-triangle_l / 2, -triangle_w / 2, 0)

    p3 = Point(motor_1[0], motor_1[1], 0)  # propeller 1
    p4 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 0), motor_1[1] + prop_r * np.cos(pi / 4 * 0), 0)
    p5 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 1), motor_1[1] + prop_r * np.cos(pi / 4 * 1), 0)
    p6 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 2), motor_1[1] + prop_r * np.cos(pi / 4 * 2), 0)
    p7 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 3), motor_1[1] + prop_r * np.cos(pi / 4 * 3), 0)
    p8 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 4), motor_1[1] + prop_r * np.cos(pi / 4 * 4), 0)
    p9 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 5), motor_1[1] + prop_r * np.cos(pi / 4 * 5), 0)
    p10 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 6), motor_1[1] + prop_r * np.cos(pi / 4 * 6), 0)
    p11 = Point(motor_1[0] + prop_r * np.sin(pi / 4 * 7), motor_1[1] + prop_r * np.cos(pi / 4 * 7), 0)

    p12 = Point(motor_2[0], motor_2[1], 0)  # propeller 2
    p13 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 0), motor_2[1] + prop_r * np.cos(pi / 4 * 0), 0)
    p14 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 1), motor_2[1] + prop_r * np.cos(pi / 4 * 1), 0)
    p15 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 2), motor_2[1] + prop_r * np.cos(pi / 4 * 2), 0)
    p16 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 3), motor_2[1] + prop_r * np.cos(pi / 4 * 3), 0)
    p17 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 4), motor_2[1] + prop_r * np.cos(pi / 4 * 4), 0)
    p18 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 5), motor_2[1] + prop_r * np.cos(pi / 4 * 5), 0)
    p19 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 6), motor_2[1] + prop_r * np.cos(pi / 4 * 6), 0)
    p20 = Point(motor_2[0] + prop_r * np.sin(pi / 4 * 7), motor_2[1] + prop_r * np.cos(pi / 4 * 7), 0)

    p21 = Point(motor_3[0], motor_3[1], 0)  # propeller 3
    p22 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 0), motor_3[1] + prop_r * np.cos(pi / 4 * 0), 0)
    p23 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 1), motor_3[1] + prop_r * np.cos(pi / 4 * 1), 0)
    p24 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 2), motor_3[1] + prop_r * np.cos(pi / 4 * 2), 0)
    p25 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 3), motor_3[1] + prop_r * np.cos(pi / 4 * 3), 0)
    p26 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 4), motor_3[1] + prop_r * np.cos(pi / 4 * 4), 0)
    p27 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 5), motor_3[1] + prop_r * np.cos(pi / 4 * 5), 0)
    p28 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 6), motor_3[1] + prop_r * np.cos(pi / 4 * 6), 0)
    p29 = Point(motor_3[0] + prop_r * np.sin(pi / 4 * 7), motor_3[1] + prop_r * np.cos(pi / 4 * 7), 0)

    p30 = Point(motor_4[0], motor_4[1], 0)  # propeller 4
    p31 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 0), motor_4[1] + prop_r * np.cos(pi / 4 * 0), 0)
    p32 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 1), motor_4[1] + prop_r * np.cos(pi / 4 * 1), 0)
    p33 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 2), motor_4[1] + prop_r * np.cos(pi / 4 * 2), 0)
    p34 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 3), motor_4[1] + prop_r * np.cos(pi / 4 * 3), 0)
    p35 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 4), motor_4[1] + prop_r * np.cos(pi / 4 * 4), 0)
    p36 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 5), motor_4[1] + prop_r * np.cos(pi / 4 * 5), 0)
    p37 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 6), motor_4[1] + prop_r * np.cos(pi / 4 * 6), 0)
    p38 = Point(motor_4[0] + prop_r * np.sin(pi / 4 * 7), motor_4[1] + prop_r * np.cos(pi / 4 * 7), 0)

    p39 = Point(0, 0, 0)
    p40 = Point(0, 0, -h)

    # fmt: off
    points = [  p0, p1, p2, # triangle body
                p3, p4, p5, # propeller 1, portion 1
                p3, p5, p6, # 2
                p3, p6, p7, # 3
                p3, p7, p8, # 4
                p3, p8, p9, # 5
                p3, p9, p10, # 6
                p3, p10, p11, # 7
                p3, p11, p4, # 8
                p12, p13, p14, # propeller 2, portion 1
                p12, p14, p15, # 2
                p12, p15, p16, # 3
                p12, p16, p17, # 4
                p12, p17, p18, # 5
                p12, p18, p19, # 6
                p12, p19, p20, # 7
                p12, p20, p13, # 8
                p21, p22, p23, # propeller 3, portion 1
                p21, p23, p24, # 2
                p21, p24, p25, # 3
                p21, p25, p26, # 4
                p21, p26, p27, # 5
                p21, p27, p28, # 6
                p21, p28, p29, # 7
                p21, p29, p22, # 8
                p30, p31, p32, # propeller 4, portion 1
                p30, p32, p33, # 2
                p30, p33, p34, # 3
                p30, p34, p35, # 4
                p30, p35, p36, # 5
                p30, p36, p37, # 6
                p30, p37, p38, # 7
                p30, p38, p31, # 8
                p39, p40, p3, # frame
                p39, p40, p12, # frame
                p39, p40, p21, # frame
                p39, p40, p30] # frame
    # fmt: on

    assert len(points) % 3 == 0, "points must be a multiple of 3"

    # Define the colors for each face of triangular mesh, rgba
    red = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    green = ColorRGBA(0.0, 1.0, 0.0, 1.0)
    blue = ColorRGBA(0.0, 0.0, 1.0, 1.0)
    yellow = ColorRGBA(1.0, 1.0, 0.0, 1.0)

    # Set mesh colors
    mesh_colors = [color for color in [green] + [yellow] * 8 * 4 + [blue] * 4]  # arrow, propellers, frame

    assert len(mesh_colors) == len(points) // 3, "mesh_colors must be a third of the length of points"

    # Define the marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.TRIANGLE_LIST
    marker.ns = f"qd_{idx}"
    marker.id = idx
    marker.action = marker.ADD
    marker.color.a = 1.0  # must be set greater than 0 to be visible. This alpha is for the whole body.
    marker.points = points
    marker.colors = mesh_colors
    marker.scale = Vector3(1, 1, 1)
    marker.pose.orientation.w = 1.0
    marker.text = f"qd_{idx}"

    return marker


if __name__ == "__main__":
    marker_array = draw_mul_qd(3)
