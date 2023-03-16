#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: autopilot.py
Date: 11/2/2022 9:28 PM
LastEditors: LI Jinjie
LastEditTime: 11/2/2022 9:28 PM
Description: file content
"""
import numpy as np
import torch
import torch.nn as nn
import math

from ..parameters import control_parameters as AP
from tools import wrap, saturate_w_float_limit
from .pid_control import PIDControl
from .pd_control import PDControl


class AtpRate(nn.Module):
    def __init__(self, num_agent: int, ts_control: float, dtype=torch.float64) -> None:
        super().__init__()
        self.G_1_T_torch = nn.Parameter(torch.tensor(AP.G_1_T), False)

        ts_factor = ts_control / 0.02  # 确保pid参数会随着仿真步长的改变而改变

        self.zero_param = nn.Parameter(torch.zeros([num_agent, 1, 1], dtype=dtype), False)

        # 角速度环，从角速度得到期望力矩
        self.Mx_from_roll_rate = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.roll_rate_kp,
            ki=AP.roll_rate_ki / ts_factor,
            kd=AP.roll_rate_kd * ts_factor,
            sigma=AP.sigma,
            limit=100,
        )
        self.My_from_pitch_rate = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.pitch_rate_kp,
            ki=AP.pitch_rate_ki / ts_factor,
            kd=AP.pitch_rate_kd * ts_factor,
            sigma=AP.sigma,
            limit=100,
        )
        self.Mz_from_yaw_rate = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.yaw_rate_kp,
            ki=AP.yaw_rate_ki / ts_factor,
            kd=AP.yaw_rate_kd * ts_factor,
            sigma=AP.sigma,
            limit=100,
        )

    def forward(self, ego_states: torch.Tensor, cmd: torch.Tensor):
        """autopilot 更新

        Args:
            ego_states: [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
             p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, xxx]
            cmd: ["airspeed_command", "course_command", "altitude_command", "phi_feedforward"]

        Returns:
            cmd: ["vn_cmd", "ve_cmd", "vd_cmd", "psi_cmd"]
        """
        vn_cmd = cmd[:, 0:1, :]
        ve_cmd = cmd[:, 1:2, :]
        vd_cmd = cmd[:, 2:3, :]
        psi_cmd = cmd[:, 3:4, :]

        vx = ego_states[:, 13:14, :]
        vy = ego_states[:, 14:15, :]
        vz = ego_states[:, 15:16, :]

        # chi = ego_states[:, 27:28, :]

        phi = ego_states[:, 6:7, :]
        theta = ego_states[:, 7:8, :]
        psi = ego_states[:, 8:9, :]

        p = ego_states[:, 19:20, :]
        q = ego_states[:, 20:21, :]
        r = ego_states[:, 21:22, :]

        # altitude = -ego_states[:, 5:6, :]
        # Va = ego_states[:, 22:23, :]

        # ----- velocity loop -----
        # horizontal
        yaw_cmd = wrap(psi_cmd, self.zero_param.data, limit=math.pi)  # 偏航角，[-pi, pi]
        c_psi_c = torch.cos(yaw_cmd)
        s_psi_c = torch.sin(yaw_cmd)

        # Input: meter / sec. Output: meter / sec^2
        acc_x_cmd = self.acc_x_from_vx(vn_cmd, vx)
        acc_y_cmd = self.acc_y_from_vy(ve_cmd, vy)
        acc_z_cmd = self.acc_z_from_vz(vd_cmd, vz) - AP.gravity

        # horizontal
        pitch_cmd = torch.atan2(-acc_x_cmd * c_psi_c - acc_y_cmd * s_psi_c, -acc_z_cmd)
        pitch_cmd = wrap(pitch_cmd, self.zero_param.data, limit=math.pi / 2.0)  # 俯仰角，[-pi/2., pi/2.]
        c_theta_c = torch.cos(pitch_cmd)
        # s_theta_c = np.sin(pitch_cmd)

        roll_cmd = torch.atan2(c_theta_c * (-acc_x_cmd * s_psi_c + acc_y_cmd * c_psi_c), -acc_z_cmd)
        roll_cmd = wrap(roll_cmd, self.zero_param.data, limit=math.pi / 2.0)  # 滚转角, 为避免奇异值现象，设为[-pi/2., pi/2.]
        c_phi_c = torch.cos(roll_cmd)
        # s_phi_c = np.sin(roll_cmd)

        # vertical
        thrust_cmd = -AP.mass * acc_z_cmd / (c_phi_c * c_theta_c)

        # ------ attitude loop ------
        # Input limit
        roll_cmd = saturate_w_float_limit(roll_cmd, -AP.roll_input_limit, AP.roll_input_limit)
        pitch_cmd = saturate_w_float_limit(pitch_cmd, -AP.pitch_input_limit, AP.pitch_input_limit)

        # Input: radians. Output: radians / sec
        roll_rate_cmd = self.roll_rate_from_roll(roll_cmd, phi)
        pitch_rate_cmd = self.pitch_rate_from_pitch(pitch_cmd, theta)
        # yaw_rate_cmd = self.yaw_rate_from_yaw.update(yaw_cmd, self.true_state.psi)
        flag_positive = (yaw_cmd >= 0) * (psi <= yaw_cmd - torch.pi)
        flag_negative = (yaw_cmd < 0) * (psi >= yaw_cmd + torch.pi)
        yaw_cmd = yaw_cmd - torch.pi * 2 * flag_positive + torch.pi * 2 * flag_negative
        yaw_rate_cmd = self.yaw_rate_from_yaw(yaw_cmd, psi)

        # ------- attitude_rate_loop --------
        torque_x_cmd = self.Mx_from_roll_rate(roll_rate_cmd, p)
        torque_y_cmd = self.My_from_pitch_rate(pitch_rate_cmd, q)
        torque_z_cmd = self.Mz_from_yaw_rate(yaw_rate_cmd, r)

        # ------- power_distribution -------
        thrust_cmd[thrust_cmd < 0] = 0

        omega_square = self.G_1_T_torch @ torch.cat((thrust_cmd, torque_x_cmd, torque_y_cmd, torque_z_cmd), 1)
        # force:N, torque:Nm
        omega_square[omega_square < 0] = 0
        delta = torch.sqrt(omega_square)

        return delta


if __name__ == "__main__":
    autopilot_func = Autopilot(0.02, 2000).to("cuda")
    autopilot = torch.jit.script(autopilot_func)

    print(1)
