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
from geometry_msgs.msg import PoseStamped
from quadrotor import MulQuadrotors


class DopQdNode:
    def __init__(self):
        rospy.init_node("dop_qd_node", anonymous=False)
        # self.num_agent = rospy.get_param("num_agent")
        self.ts_sim = rospy.get_param(rospy.get_name() + "/ts_sim")  # time step for simulation
        self.ts_ctl = rospy.get_param(rospy.get_name() + "/ts_ctl")  # time step for inner Controller
        self.ts_viz = rospy.get_param(rospy.get_name() + "/ts_viz")  # time step for visualization

        self.qd_init_states = rospy.get_param(rospy.get_name() + "/qd_init_states")

        self.num_agent = len(self.qd_init_states)

        rospy.loginfo(
            f"Load params: \n num_agent: {self.num_agent} \n sim_ts: {self.ts_sim} \n control_ts: {self.ts_ctl}"
        )

        self.pose_pub_list = []
        for i in range(self.num_agent):
            self.pose_pub_list.append(rospy.Publisher("qd_" + str(i), PoseStamped, queue_size=1))

        self.rate = rospy.Rate(1 / self.ts_sim)
        self.model = self._load_model()  # Load PyTorch model

    def run(self):
        """
        ego_states: [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
                 p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd]   len=31
        cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
        """
        ego_states = self._load_init_states()

        body_rate_cmd = torch.zeros([self.num_agent, 4, 1], dtype=torch.float64).to("cuda")
        body_rate_cmd[:, 3, 0] = 0.0  # throttle_cmd

        rospy.loginfo("Start simulation!")

        while not rospy.is_shutdown():
            time_a = time.perf_counter()

            ego_states = self._run_model(ego_states, body_rate_cmd)
            self._publish_points(ego_states)

            time_b = time.perf_counter()

            self.rate.sleep()

    def _load_init_states(self):
        ego_states = torch.zeros([self.num_agent, 31, 1], dtype=torch.float64).to("cuda")
        ego_states[:, 9, 0] = 1.0  # ew
        for i in range(self.num_agent):
            ego_states[i][3][0] = self.qd_init_states[i][0]  # n
            ego_states[i][4][0] = self.qd_init_states[i][1]  # e
            ego_states[i][5][0] = self.qd_init_states[i][2]  # d

        return ego_states

    def _load_model(self):
        # Load PyTorch model here
        mul_qd = MulQuadrotors(self.num_agent, self.ts_sim, torch.float64).requires_grad_(False)
        if torch.cuda.is_available():
            mul_qd = mul_qd.to("cuda")
        sm_mul_qd = torch.jit.script(mul_qd)  # Script model for faster inference

        rospy.loginfo("Load model successfully!")
        return sm_mul_qd

    def _run_model(self, ego_states: torch.Tensor, body_rate_cmd: torch.Tensor):
        # Run PyTorch model and get output tensor
        output_tensor = self.model(ego_states, body_rate_cmd, self.ts_sim)
        return output_tensor

    def _publish_points(self, ego_states: torch.Tensor):
        # rospy.loginfo("Publish points for one round!")

        for i in range(self.num_agent):
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = ego_states[i][4][0]  # e
            pose_msg.pose.position.y = ego_states[i][3][0]  # n
            pose_msg.pose.position.z = -ego_states[i][5][0]  # u
            pose_msg.pose.orientation.w = ego_states[i][9][0]  # ew
            pose_msg.pose.orientation.x = ego_states[i][10][0]  # ex
            pose_msg.pose.orientation.y = ego_states[i][11][0]  # ey
            pose_msg.pose.orientation.z = ego_states[i][12][0]  # ez
            self.pose_pub_list[i].publish(pose_msg)


if __name__ == "__main__":
    try:
        node = DopQdNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
