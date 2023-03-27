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
from draw_qd import MulQdDrawer


class DopQdNode:
    def __init__(self) -> None:
        rospy.init_node("dop_qd_node", anonymous=False)

        # params
        self.ts_sim = rospy.get_param(rospy.get_name() + "/ts_sim")  # time step for simulation
        self.ts_ctl = rospy.get_param(rospy.get_name() + "/ts_ctl")  # time step for inner Controller
        self.ts_viz = rospy.get_param(rospy.get_name() + "/ts_viz")  # time step for visualization

        # simulation
        qd_init_states = rospy.get_param(rospy.get_name() + "/qd_init_states")  # initial states for quadrotors
        self.num_agent = len(qd_init_states)  # number of quadrotors

        self.ego_states = self._load_init_states(qd_init_states)
        self.body_rate_cmd = self._load_init_cmd()
        self.model = self._load_model()  # Load PyTorch dynamics model

        rospy.loginfo(
            f"Load params: \n num_agent: {self.num_agent} \n sim_ts: {self.ts_sim} \n control_ts: {self.ts_ctl}"
        )

        # visualization
        self.viz_drawer = MulQdDrawer(
            self.num_agent, has_qd=True, has_text=True, has_downwash=True, has_rpm=True, max_krpm=50.0, min_krpm=0.0
        )
        self.marker_array_pub = rospy.Publisher("mul_qds_viz", MarkerArray, queue_size=5)
        self.viz_is_init = False

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

        # low_level controller and dynamics
        self.ego_states = self._run_model(self.ego_states, self.body_rate_cmd)

        time_b = time.perf_counter()
        self.time_guarder.measure_run_t(time_b - time_a)

    def pub_viz_callback(self, timer: rospy.timer.TimerEvent) -> None:
        # rospy.loginfo("Publish points for one round!")
        viz_marker_array = self.viz_drawer.update(self.ego_states)
        self.marker_array_pub.publish(viz_marker_array)

    def _load_init_states(self, qd_init_states: list):
        num_agent = self.num_agent
        ego_states = torch.zeros([num_agent, 35, 1], dtype=torch.float64).to("cuda")
        ego_states[:, 9, 0] = 1.0  # ew
        for i in range(num_agent):
            ego_states[i][3][0] = qd_init_states[i][0]  # e
            ego_states[i][4][0] = qd_init_states[i][1]  # n
            ego_states[i][5][0] = qd_init_states[i][2]  # u

        return ego_states

    def _load_init_cmd(self):
        body_rate_cmd = torch.zeros([self.num_agent, 4, 1], dtype=torch.float64).to("cuda")
        body_rate_cmd[:, 3, 0] = 0.235  # throttle_cmd
        body_rate_cmd[0, 0, 0] = 0.1  # roll_rate_cmd
        body_rate_cmd[1, 1, 0] = 0.1  # pitch_rate_cmd
        body_rate_cmd[2, 2, 0] = 0.5  # yaw_rate_cmd

        return body_rate_cmd

    def _load_model(self) -> torch.jit.ScriptModule:
        # Load PyTorch model here
        mul_qd = MulQuadrotors(self.num_agent, self.ts_sim, torch.float64).requires_grad_(False)
        if torch.cuda.is_available():
            mul_qd = mul_qd.to("cuda")
        sm_mul_qd = torch.jit.script(mul_qd)  # Script model for faster inference
        rospy.loginfo("Load model successfully!")

        # set initial ego_states and body_rate_cmd
        self.model = sm_mul_qd
        self.ego_states = self._run_model(self.ego_states, self.body_rate_cmd)

        return sm_mul_qd

    def _run_model(self, ego_states: torch.Tensor, body_rate_cmd: torch.Tensor) -> torch.Tensor:
        # Run PyTorch model and get output tensor
        new_ego_states = self.model(self.ts_sim, ego_states, body_rate_cmd)
        return new_ego_states


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
