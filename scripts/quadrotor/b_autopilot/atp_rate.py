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
import torch
import torch.nn as nn

from ..params import control_param as CP
from .pid_control import PIDControl


class AtpRate(nn.Module):
    def __init__(self, num_agent: int, ts_control: float, has_battery: bool, dtype=torch.float64) -> None:
        super().__init__()
        self.G_1_inv_torch = nn.Parameter(torch.tensor(CP.G_1_inv), False)

        ts_factor = ts_control / 0.02  # 确保pid参数会随着仿真步长的改变而改变

        self.zero_param = nn.Parameter(torch.zeros([num_agent, 1, 1], dtype=dtype), False)

        self.has_battery = has_battery

        # 角速度环，从角速度得到期望力矩
        self.Mx_from_roll_rate = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=CP.roll_rate_kp,
            ki=CP.roll_rate_ki / ts_factor,
            kd=CP.roll_rate_kd * ts_factor,
            sigma=CP.sigma,
            u_limit=999.0,
        )
        self.My_from_pitch_rate = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=CP.pitch_rate_kp,
            ki=CP.pitch_rate_ki / ts_factor,
            kd=CP.pitch_rate_kd * ts_factor,
            sigma=CP.sigma,
            u_limit=999.0,
        )
        self.Mz_from_yaw_rate = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=CP.yaw_rate_kp,
            ki=CP.yaw_rate_ki / ts_factor,
            kd=CP.yaw_rate_kd * ts_factor,
            sigma=CP.sigma,
            u_limit=999.0,
        )

    def forward(self, ego_states: torch.Tensor, cmd: torch.Tensor, all_sim_t: float):
        """autopilot 更新

        Args:
            ego_states: [player, group, HP, east, north, up, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
             p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, xxx]
            cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
            all_sim_t: float, all simulation time

            rate_cmd is in rad/s.
            Note that throttle_cmd is a [0, 1] normalized float value.

        Returns:
            cmd: [o1, o2, o3, o4], kRPM
        """
        roll_rate_cmd = cmd[:, 0:1, :]
        pitch_rate_cmd = cmd[:, 1:2, :]
        yaw_rate_cmd = cmd[:, 2:3, :]
        throttle_cmd = cmd[:, 3:4, :]

        p = ego_states[:, 19:20, :]
        q = ego_states[:, 20:21, :]
        r = ego_states[:, 21:22, :]

        # throttle
        if self.has_battery:
            voltage = 4.2 - all_sim_t / CP.t_all * (4.2 - 3.6)
            voltage_cf = voltage / 4.2
        else:
            voltage_cf = 1.0

        thrust_cmd = 4 * (throttle_cmd * CP.k_th + CP.b_th) * voltage_cf

        thrust_cmd[thrust_cmd < 0] = 0

        # ------- attitude_rate_loop --------
        torque_x_cmd = self.Mx_from_roll_rate(roll_rate_cmd, p)
        torque_y_cmd = self.My_from_pitch_rate(pitch_rate_cmd, q)
        torque_z_cmd = self.Mz_from_yaw_rate(yaw_rate_cmd, r)

        # ------- power_distribution -------
        thrust_cmd[thrust_cmd < 0] = 0

        thrust_per_motor = self.G_1_inv_torch @ torch.cat((thrust_cmd, torque_x_cmd, torque_y_cmd, torque_z_cmd), 1)
        # force:N, torque:Nm

        thrust_per_motor[thrust_per_motor < 0] = 0

        # thrust_per_motor = ct * omega ** 2
        delta = torch.sqrt(thrust_per_motor / CP.k_t)

        return delta


if __name__ == "__main__":
    autopilot_func = AtpRate(2000, 0.02).to("cuda")
    autopilot = torch.jit.script(autopilot_func)

    print(1)
