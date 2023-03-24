#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: dop_qd_node.py
Date: 2023/3/21 下午3:58
Description:
"""

import sys
import os

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)

import copy
import time
import rospy
import torch
from geometry_msgs.msg import Pose
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
        self.ego_states = self._load_init_states(self.num_agent, qd_init_states)

        rospy.loginfo(
            f"Load params: \n num_agent: {self.num_agent} \n sim_ts: {self.ts_sim} \n control_ts: {self.ts_ctl}"
        )

        # visualization
        self.viz_marker_array = draw_mul_qd(self.num_agent)  # viz, text, downwash
        self.marker_array_pub = rospy.Publisher("mul_qds_viz", MarkerArray, queue_size=5)
        self.viz_is_init = False

        # simulation
        self.ctl_time = rospy.Time.now()
        self.model = self._load_model()  # Load PyTorch model

        # init cmd
        body_rate_cmd = torch.zeros([self.num_agent, 4, 1], dtype=torch.float64).to("cuda")
        body_rate_cmd[:, 3, 0] = 0.235  # throttle_cmd
        body_rate_cmd[0, 0, 0] = 0.1  # roll_rate_cmd
        body_rate_cmd[1, 1, 0] = 0.1  # pitch_rate_cmd
        body_rate_cmd[2, 2, 0] = 0.1  # yaw_rate_cmd
        self.body_rate_cmd = body_rate_cmd

        # timer for various frequencies  # TODO: delete this
        self.time_guarder = TimeGuarder(ts_sim=self.ts_sim, ts_measure=5)

        # register various timers
        # note that the callback function will be passed a rospy.timer.TimerEvent object after self
        rospy.Timer(rospy.Duration(self.ts_sim), self.sim_loop_callback)
        rospy.Timer(rospy.Duration(self.ts_viz), self.pub_viz_callback)

        rospy.loginfo("Start simulation!")

    def sim_loop_callback(self, timer: rospy.timer.TimerEvent) -> None:
        """
        ego_states: [player, group, HP, east, north, up, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
                 p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd]   len=31
        cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
        """

        time_a = time.perf_counter()

        # low-level controller
        if (rospy.Time.now() - self.ctl_time).to_sec() > self.ts_ctl:
            # run model with low_level controller
            self.ctl_time = rospy.Time.now()

        # low_level controller and dynamics
        self.ego_states = self._run_model(self.ego_states, self.body_rate_cmd)

        time_b = time.perf_counter()
        self.time_guarder.measure_run_t(time_b - time_a)

    def pub_viz_callback(self, timer: rospy.timer.TimerEvent) -> None:
        # rospy.loginfo("Publish points for one round!")
        ego_states = self.ego_states

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
            viz_marker = self.viz_marker_array.markers[i]
            viz_marker.header.stamp = rospy.Time.now()
            if not self.viz_is_init:
                viz_marker.action = viz_marker.ADD
            else:
                viz_marker.action = viz_marker.MODIFY
            viz_marker.pose = ego_pose

            # text
            text_marker = self.viz_marker_array.markers[i + self.num_agent]
            text_marker.header.stamp = rospy.Time.now()
            if not self.viz_is_init:
                text_marker.action = text_marker.ADD
            else:
                text_marker.action = text_marker.MODIFY
            text_marker.pose = copy.deepcopy(ego_pose)
            text_marker.pose.position.z += 0.1  # bias for text

            # downwash
            downwash_marker = self.viz_marker_array.markers[i + 2 * self.num_agent]
            downwash_marker.header.stamp = rospy.Time.now()
            if not self.viz_is_init:
                downwash_marker.action = downwash_marker.ADD
            else:
                downwash_marker.action = downwash_marker.MODIFY
            downwash_marker.pose = copy.deepcopy(ego_pose)
            downwash_marker.pose.position.z -= 0.6  # bias for downwash

            if not self.viz_is_init:
                self.viz_is_init = True

        self.marker_array_pub.publish(self.viz_marker_array)

    @staticmethod
    def _load_init_states(num_agent: float, qd_init_states: list) -> torch.Tensor:
        ego_states = torch.zeros([num_agent, 31, 1], dtype=torch.float64).to("cuda")
        ego_states[:, 9, 0] = 1.0  # ew
        for i in range(num_agent):
            ego_states[i][3][0] = qd_init_states[i][0]  # e
            ego_states[i][4][0] = qd_init_states[i][1]  # n
            ego_states[i][5][0] = qd_init_states[i][2]  # u

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


class TimeGuarder:
    def __init__(self, ts_sim: float, ts_measure: float = 1):
        self.measure_round = ts_measure / ts_sim
        self.ts_sim = ts_sim

        self.run_round = 0
        self.run_t = 0.0
        self.clear_run_t()

    def clear_run_t(self):
        self.run_round = 0
        self.run_t = 0.0

    def measure_run_t(self, t_one_round: float):
        self.run_t += t_one_round
        self.run_round += 1
        if self.run_round == self.measure_round:
            if self.ts_sim < self.run_t / self.measure_round:
                rospy.logwarn(
                    f"Simulation is too slow! ts_sim: {self.ts_sim * 1000:.3f} ms < ts_one_round: {self.run_t / self.measure_round * 1000:.3f} ms"
                )

            # # DEBUG only
            # rospy.loginfo(
            #     f"Average running time for {self.measure_round} rounds: {self.run_t / self.measure_round * 1000:.3f} ms \n"
            #     f"Time step for simulation is {self.ts_sim * 1000:.3f} ms \n"
            #     f"Real time simulation is {self.ts_sim > self.run_t / self.measure_round} !"
            # )

            self.clear_run_t()


if __name__ == "__main__":
    try:
        node = DopQdNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
