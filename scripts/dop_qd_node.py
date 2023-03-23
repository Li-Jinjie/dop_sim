#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: dop_qd_node.py
Date: 2023/3/21 下午3:58
Description: 
"""
import sys
import time
from os.path import expanduser

home = expanduser("~")
sys.path.append(home + "/ljj_ws/src/dop_qd_sim/scripts")
# sys.path.append(home + "/catkin_ws/src/pc/min_snap_traj_gen/scripts")


import rospy
import torch
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from quadrotor import MulQuadrotors
from draw_qd import draw_mul_qd


class DopQdNode:
    def __init__(self) -> None:
        rospy.init_node("dop_qd_node", anonymous=False)

        # params
        self.ts_sim = rospy.get_param(rospy.get_name() + "/ts_sim")  # time step for simulation
        self.ts_ctl = rospy.get_param(rospy.get_name() + "/ts_ctl")  # time step for inner Controller
        self.ts_viz = rospy.get_param(rospy.get_name() + "/ts_viz")  # time step for visualization

        qd_init_states = rospy.get_param(rospy.get_name() + "/qd_init_states")  # initial states for quadrotors
        self.num_agent = len(qd_init_states)  # number of quadrotors
        self.init_ego_states = self._load_init_states(self.num_agent, qd_init_states)

        rospy.loginfo(
            f"Load params: \n num_agent: {self.num_agent} \n sim_ts: {self.ts_sim} \n control_ts: {self.ts_ctl}"
        )

        # visualization
        self.mul_qd_marker_array = draw_mul_qd(self.num_agent)
        self.marker_array_pub = rospy.Publisher("mul_qds_viz", MarkerArray, queue_size=5)
        self.viz_time = rospy.Time.now()  # the time of the last visualization
        self._pub_viz(self.init_ego_states, is_first=True)

        # simulation
        self.rate = rospy.Rate(1 / self.ts_sim)
        self.model = self._load_model()  # Load PyTorch model

    def run(self) -> None:
        """
        ego_states: [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
                 p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd]   len=31
        cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
        """
        ego_states = self.init_ego_states

        body_rate_cmd = torch.zeros([self.num_agent, 4, 1], dtype=torch.float64).to("cuda")
        body_rate_cmd[:, 3, 0] = 0.0  # throttle_cmd

        rospy.loginfo("Start simulation!")

        while not rospy.is_shutdown():
            # time_a = time.perf_counter()

            ego_states = self._run_model(ego_states, body_rate_cmd)

            if (rospy.Time.now() - self.viz_time).to_sec() > self.ts_viz:
                self._pub_viz(ego_states)
                self.viz_time = rospy.Time.now()

            # time_b = time.perf_counter()

            self.rate.sleep()

    @staticmethod
    def _load_init_states(num_agent: float, qd_init_states: list) -> torch.Tensor:
        ego_states = torch.zeros([num_agent, 31, 1], dtype=torch.float64).to("cuda")
        ego_states[:, 9, 0] = 1.0  # ew
        for i in range(num_agent):
            ego_states[i][3][0] = qd_init_states[i][0]  # n
            ego_states[i][4][0] = qd_init_states[i][1]  # e
            ego_states[i][5][0] = qd_init_states[i][2]  # d

        return ego_states

    def _load_model(self) -> torch.jit.ScriptModule:
        # Load PyTorch model here
        mul_qd = MulQuadrotors(self.num_agent, self.ts_sim, torch.float64).requires_grad_(False)
        if torch.cuda.is_available():
            mul_qd = mul_qd.to("cuda")
        sm_mul_qd = torch.jit.script(mul_qd)  # Script model for faster inference

        rospy.loginfo("Load model successfully!")
        return sm_mul_qd

    def _run_model(self, ego_states: torch.Tensor, body_rate_cmd: torch.Tensor) -> torch.Tensor:
        # Run PyTorch model and get output tensor
        output_tensor = self.model(ego_states, body_rate_cmd, self.ts_sim)
        return output_tensor

    def _pub_viz(self, ego_states: torch.Tensor, is_first=False) -> None:
        rospy.loginfo("Publish points for one round!")

        for i, marker in enumerate(self.mul_qd_marker_array.markers):

            marker.header.stamp = rospy.Time.now()
            if is_first:
                marker.action = marker.ADD
            else:
                marker.action = marker.MODIFY

            marker.pose.position.x = ego_states[i][4][0]  # e
            marker.pose.position.y = ego_states[i][3][0]  # n
            marker.pose.position.z = -ego_states[i][5][0]  # u
            marker.pose.orientation.w = ego_states[i][9][0]  # ew
            marker.pose.orientation.x = ego_states[i][10][0]  # ex
            marker.pose.orientation.y = ego_states[i][11][0]  # ey
            marker.pose.orientation.z = ego_states[i][12][0]  # ez

        self.marker_array_pub.publish(self.mul_qd_marker_array)


if __name__ == "__main__":
    try:
        node = DopQdNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
