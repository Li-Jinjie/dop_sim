#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: draw_qd.py
Date: 2023/3/23 上午9:31
Description:
"""
import copy
import numpy as np
import rospy
import torch
import matplotlib.pyplot as plt

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose, Point, Vector3


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


def draw_one_qd(idx: int) -> Marker:
    # points are in FLU coordinates
    p = [Point()] * 41
    p[0] = Point(triangle_l / 2, 0, 0)  # triangle on the body
    p[1] = Point(-triangle_l / 2, triangle_w / 2, 0)
    p[2] = Point(-triangle_l / 2, -triangle_w / 2, 0)

    p[3] = Point(motor_1[0], motor_1[1], 0)  # propeller 1
    for i in range(8):
        p[4 + i] = Point(motor_1[0] + prop_r * np.sin(pi / 4 * i), motor_1[1] + prop_r * np.cos(pi / 4 * i), 0)

    p[12] = Point(motor_2[0], motor_2[1], 0)  # propeller 2
    for i in range(8):
        p[13 + i] = Point(motor_2[0] + prop_r * np.sin(pi / 4 * i), motor_2[1] + prop_r * np.cos(pi / 4 * i), 0)

    p[21] = Point(motor_3[0], motor_3[1], 0)  # propeller 3
    for i in range(8):
        p[22 + i] = Point(motor_3[0] + prop_r * np.sin(pi / 4 * i), motor_3[1] + prop_r * np.cos(pi / 4 * i), 0)

    p[30] = Point(motor_4[0], motor_4[1], 0)  # propeller 4
    for i in range(8):
        p[31 + i] = Point(motor_4[0] + prop_r * np.sin(pi / 4 * i), motor_4[1] + prop_r * np.cos(pi / 4 * i), 0)

    # frame
    p[39] = Point(0, 0, 0)
    p[40] = Point(0, 0, -h)

    # fmt: off
    points = [  p[0], p[1], p[2], # triangle body
                p[3], p[4], p[5], # propeller 1, portion 1
                p[3], p[5], p[6], # 2
                p[3], p[6], p[7], # 3
                p[3], p[7], p[8], # 4
                p[3], p[8], p[9], # 5
                p[3], p[9], p[10], # 6
                p[3], p[10], p[11], # 7
                p[3], p[11], p[4], # 8
                p[12], p[13], p[14], # propeller 2, portion 1
                p[12], p[14], p[15], # 2
                p[12], p[15], p[16], # 3
                p[12], p[16], p[17], # 4
                p[12], p[17], p[18], # 5
                p[12], p[18], p[19], # 6
                p[12], p[19], p[20], # 7
                p[12], p[20], p[13], # 8
                p[21], p[22], p[23], # propeller 3, portion 1
                p[21], p[23], p[24], # 2
                p[21], p[24], p[25], # 3
                p[21], p[25], p[26], # 4
                p[21], p[26], p[27], # 5
                p[21], p[27], p[28], # 6
                p[21], p[28], p[29], # 7
                p[21], p[29], p[22], # 8
                p[30], p[31], p[32], # propeller 4, portion 1
                p[30], p[32], p[33], # 2
                p[30], p[33], p[34], # 3
                p[30], p[34], p[35], # 4
                p[30], p[35], p[36], # 5
                p[30], p[36], p[37], # 6
                p[30], p[37], p[38], # 7]
                p[30], p[38], p[31], # 8
                p[39], p[40], p[3], # frame
                p[39], p[40], p[12], # frame
                p[39], p[40], p[21], # frame
                p[39], p[40], p[30]] # frame
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


def draw_one_text(idx: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.TEXT_VIEW_FACING
    marker.ns = f"qd_{idx}"
    marker.id = idx + 900000
    marker.action = marker.ADD
    marker.color = ColorRGBA(
        1.0, 0.39, 0.28, 1.0
    )  # must be set greater than 0 to be visible. This alpha is for the whole body.
    marker.text = f"qd_{idx}"
    marker.scale.z = 0.1
    marker.pose.orientation.w = 1.0

    return marker


def draw_one_downwash(idx: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.CYLINDER
    marker.ns = f"qd_{idx}"
    marker.id = idx + 800000
    marker.action = marker.ADD
    marker.color = ColorRGBA(92 / 255, 179 / 255, 204 / 255, 0.05)  # 碧青
    marker.scale = Vector3(0.5, 0.5, 1.0)
    marker.pose.orientation.w = 1.0

    return marker


class MulQdDrawer:
    def __init__(
        self,
        num_agent: int,
        has_qd: bool = True,
        has_text: bool = True,
        has_downwash: bool = True,
        has_rpm: bool = False,
        max_krpm: float = 30.0,
        min_krpm: float = 0.0,
    ):
        self.num_agent = num_agent

        self.has_qd = has_qd
        self.has_text = has_text
        self.has_downwash = has_downwash
        self.has_rpm = has_rpm

        self.viz_marker_array = self.draw_mul_qd(num_agent)  # viz, text, downwash
        self.viz_is_init = False

        if self.has_rpm:
            self.max_rpm = max_krpm
            self.min_rpm = min_krpm
            self.cmap = plt.get_cmap("YlOrRd")

    def draw_mul_qd(self, num_agent: int) -> MarkerArray:
        marker_array = MarkerArray()

        # viz
        if self.has_qd:
            for i in range(num_agent):
                marker = draw_one_qd(i)
                marker_array.markers.append(marker)

        # text
        if self.has_text:
            for i in range(num_agent):
                text = draw_one_text(i)
                marker_array.markers.append(text)

        # downwash
        if self.has_downwash:
            for i in range(num_agent):
                downwash = draw_one_downwash(i)
                marker_array.markers.append(downwash)

        return marker_array

    def update(self, ego_states: torch.Tensor) -> MarkerArray:
        # viz
        for i in range(self.num_agent):
            ego_pose = Pose()
            ego_pose.position.x = ego_states[i][3][0]  # e
            ego_pose.position.y = ego_states[i][4][0]  # n
            ego_pose.position.z = ego_states[i][5][0]  # u
            ego_pose.orientation.w = ego_states[i][9][0]  # ew
            ego_pose.orientation.x = ego_states[i][10][0]  # ex
            ego_pose.orientation.y = ego_states[i][11][0]  # ey
            ego_pose.orientation.z = ego_states[i][12][0]  # ez

            # viz
            if self.has_qd:
                viz_marker = self.viz_marker_array.markers[i]
                viz_marker.header.stamp = rospy.Time.now()
                if not self.viz_is_init:
                    viz_marker.action = viz_marker.ADD
                else:
                    viz_marker.action = viz_marker.MODIFY
                viz_marker.pose = ego_pose

                if self.has_rpm:
                    for j in range(4):
                        rpm = ego_states[i][31 + j][0].cpu()
                        rpm_norm = (rpm - self.min_rpm) / (self.max_rpm - self.min_rpm)
                        rgba = self.cmap(rpm_norm)
                        start_index = j * 8 + 1
                        end_index = (j + 1) * 8 + 1
                        viz_marker.colors[start_index:end_index] = [ColorRGBA(rgba[0], rgba[1], rgba[2], rgba[3])] * 8

            # text
            if self.has_text:
                text_marker = self.viz_marker_array.markers[i + self.num_agent * self.has_qd]
                text_marker.header.stamp = rospy.Time.now()
                if not self.viz_is_init:
                    text_marker.action = text_marker.ADD
                else:
                    text_marker.action = text_marker.MODIFY
                text_marker.pose = copy.deepcopy(ego_pose)
                text_marker.pose.position.z += 0.1  # bias for text

            # downwash
            if self.has_downwash:
                downwash_marker = self.viz_marker_array.markers[
                    i + self.num_agent * self.has_qd + self.num_agent * self.has_text
                ]
                downwash_marker.header.stamp = rospy.Time.now()
                if not self.viz_is_init:
                    downwash_marker.action = downwash_marker.ADD
                else:
                    downwash_marker.action = downwash_marker.MODIFY
                downwash_marker.pose = copy.deepcopy(ego_pose)
                downwash_marker.pose.position.z -= 0.6  # bias for downwash

            if not self.viz_is_init:
                self.viz_is_init = True

        return self.viz_marker_array
