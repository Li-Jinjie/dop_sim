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

import time
import rospy
import torch
from visualization_msgs.msg import MarkerArray
from mavros_msgs.msg import AttitudeTarget, State, ESCStatus, ESCStatusItem
from nav_msgs.msg import Odometry
from quadrotor import MulQuadrotors
from draw_qd import MulQdDrawer


class DopQdNode:
    def __init__(self) -> None:
        rospy.init_node("dop_qd_node", anonymous=False)

        # params
        self.ts_sim = rospy.get_param(rospy.get_name() + "/ts_sim")  # time step for simulation
        self.ts_ctl = rospy.get_param(rospy.get_name() + "/ts_ctl")  # time step for inner Controller
        self.ts_viz = rospy.get_param(rospy.get_name() + "/ts_viz")  # time step for visualization
        self.has_downwash = rospy.get_param(rospy.get_name() + "/has_downwash")
        self.has_motor_model = rospy.get_param(rospy.get_name() + "/has_motor_model")
        self.has_battery = rospy.get_param(rospy.get_name() + "/has_battery")

        params_log = {
            "ts_sim": "ts_sim [s]",
            "ts_ctl": "ts_control for body rate controller [s]",
            "ts_viz": "ts_viz [s]",
            "has_downwash": "has_downwash",
            "has_motor_model": "has_motor_model",
            "has_battery": "has_battery",
        }  # param_name: explanation, use for

        rospy.loginfo("Load params: \n " + "\n ".join(f"- {params_log[k]}: {self.__dict__[k]}" for k in params_log))

        # simulation
        qd_init_states = rospy.get_param(rospy.get_name() + "/qd_init_states")  # initial states for quadrotors
        self.num_agent = len(qd_init_states)  # number of quadrotors

        self.ego_states, self.ego_names = self._load_init_states(qd_init_states)
        rospy.loginfo(f"Load {self.num_agent} quadrotors: {self.ego_names}")
        self.body_rate_cmd = self._load_init_cmd()
        self.model = self._load_model()  # Load PyTorch dynamics model

        # control related states
        # - odometry
        self.mul_odom = [Odometry()] * self.num_agent
        self.mul_odom_pub = []
        for i in range(self.num_agent):
            self.mul_odom_pub.append(
                rospy.Publisher(f"/{self.ego_names[i]}/mavros/local_position/odom", Odometry, queue_size=5)
            )

        # - ESC status, mainly rpm
        esc_status = ESCStatus()
        for i in range(4):  # quadrotor
            esc_status.esc_status.append(ESCStatusItem())
        self.mul_esc = [esc_status] * self.num_agent
        self.mul_esc_pub = []
        for i in range(self.num_agent):
            self.mul_esc_pub.append(rospy.Publisher(f"/{self.ego_names[i]}/mavros/esc_status", ESCStatus, queue_size=5))

        # - state
        state = State()
        state.mode = "OFFBOARD"
        state.armed = True
        state.connected = True
        self.mul_state = [state] * self.num_agent
        self.mul_state_pub = []
        for i in range(self.num_agent):
            self.mul_state_pub.append(rospy.Publisher(f"/{self.ego_names[i]}/mavros/state", State, queue_size=5))

        # visualization
        self.viz_drawer = MulQdDrawer(
            self.num_agent, self.ego_names, True, True, self.has_downwash, has_rpm=True, max_krpm=24.0, min_krpm=0.0
        )
        self.marker_array_pub = rospy.Publisher("mul_qds_viz", MarkerArray, queue_size=5)
        self.viz_is_init = False

        # register various timers
        # note that the callback function will be passed a rospy.timer.TimerEvent object after self
        self.tmr_sim = rospy.Timer(rospy.Duration(self.ts_sim), self.sim_loop_callback)
        self.tmr_viz = rospy.Timer(rospy.Duration(self.ts_viz), self.pub_viz_callback)
        self.tmr_pub_odom = rospy.Timer(rospy.Duration(1 / 100), self.pub_odom_callback)
        self.tmr_pub_esc = rospy.Timer(rospy.Duration(1 / 50), self.pub_esc_callback)
        self.tmr_pub_px4_state = rospy.Timer(rospy.Duration(1), self.pub_state_callback)

        # register subscriber
        for i in range(self.num_agent):
            rospy.Subscriber(
                f"/{self.ego_names[i]}/mavros/setpoint_raw/attitude", AttitudeTarget, self.sub_body_rate_cmd_cb, i
            )

        rospy.loginfo("Start simulation!")

    def sim_loop_callback(self, timer: rospy.timer.TimerEvent) -> None:
        """
        ego_states: [player, group, HP, east, north, up, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
                 p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd]   len=31
        cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
        """
        # ----- check if simulation is too slow -----
        if timer.last_duration is not None and self.ts_sim < timer.last_duration:
            rospy.logwarn(
                f"Simulation is too slow!"
                f"ts_sim: {self.ts_sim * 1000:.3f} ms < ts_one_round: {timer.last_duration * 1000:.3f} ms"
            )
        # ------------------------------------------

        # low_level controller and dynamics
        self.ego_states = self._run_model(self.ego_states, self.body_rate_cmd)

    def pub_viz_callback(self, timer: rospy.timer.TimerEvent) -> None:
        # rospy.loginfo("Publish points for one round!")
        viz_marker_array = self.viz_drawer.update(self.ego_states)
        self.marker_array_pub.publish(viz_marker_array)

    def pub_odom_callback(self, timer: rospy.timer.TimerEvent) -> None:
        for i in range(self.num_agent):
            self.mul_odom[i].header.stamp = rospy.Time.now()
            self.mul_odom[i].header.frame_id = "odom"
            self.mul_odom[i].pose.pose.position.x = self.ego_states[i][3][0]  # e
            self.mul_odom[i].pose.pose.position.y = self.ego_states[i][4][0]  # n
            self.mul_odom[i].pose.pose.position.z = self.ego_states[i][5][0]  # u
            self.mul_odom[i].pose.pose.orientation.x = self.ego_states[i][10][0]  # ex
            self.mul_odom[i].pose.pose.orientation.y = self.ego_states[i][11][0]  # ey
            self.mul_odom[i].pose.pose.orientation.z = self.ego_states[i][12][0]  # ez
            self.mul_odom[i].pose.pose.orientation.w = self.ego_states[i][9][0]  # ew
            self.mul_odom[i].twist.twist.linear.x = self.ego_states[i][13][0]  # vx
            self.mul_odom[i].twist.twist.linear.y = self.ego_states[i][14][0]  # vy
            self.mul_odom[i].twist.twist.linear.z = self.ego_states[i][15][0]  # vz
            self.mul_odom[i].twist.twist.angular.x = self.ego_states[i][19][0]  # p
            self.mul_odom[i].twist.twist.angular.y = self.ego_states[i][20][0]  # q
            self.mul_odom[i].twist.twist.angular.z = self.ego_states[i][21][0]  # r
            self.mul_odom_pub[i].publish(self.mul_odom[i])

    def pub_esc_callback(self, timer: rospy.timer.TimerEvent) -> None:
        for i in range(self.num_agent):
            self.mul_esc[i].header.stamp = rospy.Time.now()
            for j in range(4):  # quadrotor
                self.mul_esc[i].esc_status[j].rpm = int(1000 * self.ego_states[i][31 + j][0])
            self.mul_esc_pub[i].publish(self.mul_esc[i])

    def pub_state_callback(self, timer: rospy.timer.TimerEvent) -> None:
        for i in range(self.num_agent):
            self.mul_state[i].header.stamp = rospy.Time.now()
            self.mul_state_pub[i].publish(self.mul_state[i])

    def sub_body_rate_cmd_cb(self, msg: AttitudeTarget, i: int) -> None:
        self.body_rate_cmd[i][0][0] = msg.body_rate.x
        self.body_rate_cmd[i][1][0] = msg.body_rate.y
        self.body_rate_cmd[i][2][0] = msg.body_rate.z
        self.body_rate_cmd[i][3][0] = msg.thrust

    def _load_init_states(self, qd_init_states: list):
        num_agent = self.num_agent

        ego_names = []

        ego_states = torch.zeros([num_agent, 35, 1], dtype=torch.float64).to("cuda")
        ego_states[:, 9, 0] = -1.0  # ew  -1.0 is compatible with PX4. If use 1.0, the quadrotor will rotate.
        ego_states[:, 31:35, :] = 8  # omega, kRPM
        for i in range(num_agent):
            if "name" in qd_init_states[i]:
                ego_names.append(qd_init_states[i]["name"])
            else:
                ego_names.append(f"qd_{i}")

            ego_states[i][3][0] = qd_init_states[i]["init_pos"][0]  # e
            ego_states[i][4][0] = qd_init_states[i]["init_pos"][1]  # n
            ego_states[i][5][0] = qd_init_states[i]["init_pos"][2]  # u

        return ego_states, ego_names

    def _load_init_cmd(self):
        body_rate_cmd = torch.zeros([self.num_agent, 4, 1], dtype=torch.float64).to("cuda")

        # Add realsense and gps modules: 1.5344 kg -> 0.23202; pure aircraft: 1.4844 kg -> 0.22400
        body_rate_cmd[:, 3, 0] = 0.283  # throttle_cmd
        # body_rate_cmd[0, 0, 0] = 0.1  # roll_rate_cmd
        # body_rate_cmd[1, 1, 0] = 0.0  # pitch_rate_cmd
        # body_rate_cmd[2, 2, 0] = 0.5  # yaw_rate_cmd
        # body_rate_cmd[3, 1, 0] = -0.05

        return body_rate_cmd

    def _load_model(self) -> torch.jit.ScriptModule:
        # Load PyTorch model here
        mul_qd = MulQuadrotors(
            self.num_agent,
            self.ts_sim,
            self.ts_ctl,
            torch.float64,
            self.has_downwash,
            self.has_motor_model,
            self.has_battery,
        ).requires_grad_(False)
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


if __name__ == "__main__":
    try:
        node = DopQdNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
