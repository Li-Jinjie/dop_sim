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

from ..params import control_param as AP
from ..tools.wrap import wrap
from ..tools.saturate import saturate_w_float_limit
from .pid_control import PIDControl


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
            cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]

            rate_cmd is in rad/s.
            Note that throttle_cmd is a [0, 1] normalized float value.

        Returns:
            cmd: [o1, o2, o3, o4], RPM
        """
        roll_rate_cmd = cmd[:, 0:1, :]
        pitch_rate_cmd = cmd[:, 1:2, :]
        yaw_rate_cmd = cmd[:, 2:3, :]
        throttle_cmd = cmd[:, 3:4, :]

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

        # vertical
        thrust_cmd = throttle_cmd * AP.collective_f_max

        # ------- attitude_rate_loop --------
        torque_x_cmd = self.Mx_from_roll_rate(roll_rate_cmd, p)
        torque_y_cmd = self.My_from_pitch_rate(pitch_rate_cmd, q)
        torque_z_cmd = self.Mz_from_yaw_rate(yaw_rate_cmd, r)

        # ------- power_distribution -------
        thrust_cmd[thrust_cmd < 0] = 0

        thrust_per_motor = self.G_1_T_torch @ torch.cat((thrust_cmd, torque_x_cmd, torque_y_cmd, torque_z_cmd), 1)
        # force:N, torque:Nm

        # thrust_per_motor = ct * omega ** 2
        thrust_per_motor[thrust_per_motor < 0] = 0

        delta = torch.sqrt(thrust_per_motor / AP.k_t)

        return delta


if __name__ == "__main__":
    autopilot_func = AtpRate(2000, 0.02).to("cuda")
    autopilot = torch.jit.script(autopilot_func)

    print(1)
