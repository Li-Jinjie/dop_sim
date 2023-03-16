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
import math

from ..params import control_param as AP
from scripts.quadrotor.tools.wrap import wrap
from scripts.quadrotor.tools.saturate import saturate_w_float_limit
from .pid_control import PIDControl
from .pd_control import PDControl


class Autopilot(nn.Module):
    def __init__(self, num_agent: int, ts_control: float, dtype=torch.float64) -> None:
        super().__init__()
        self.G_1_T_torch = nn.Parameter(torch.tensor(AP.G_1_T), False)

        ts_factor = ts_control / 0.02  # 确保pid参数会随着仿真步长的改变而改变

        self.zero_param = nn.Parameter(torch.zeros([num_agent, 1, 1], dtype=dtype), False)

        # # instantiate lateral controllers
        # self.vx_from_pn = PDControl(
        #     num_agent,
        #     Ts=ts_control,
        #     dtype=dtype,
        #     kp=AP.pn_kp,
        #     kd=AP.pn_kd * ts_factor,
        #     sigma=AP.sigma,
        #     limit=AP.vx_sat_limit,
        # )
        #
        # self.vy_from_pe = PDControl(
        #     num_agent,
        #     Ts=ts_control,
        #     dtype=dtype,
        #     kp=AP.pe_kp,
        #     kd=AP.pe_kd * ts_factor,
        #     sigma=AP.sigma,
        #     limit=AP.vy_sat_limit,
        # )
        # self.vz_from_pd = PDControl(
        #     num_agent,
        #     Ts=ts_control,
        #     dtype=dtype,
        #     kp=AP.pd_kp,
        #     kd=AP.pd_kd * ts_factor,
        #     sigma=AP.sigma,
        #     limit=AP.vz_sat_limit,
        # )

        # 速度环，从速度到期望角度和拉力
        self.acc_x_from_vx = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.vx_kp,
            ki=AP.vx_ki / ts_factor,
            kd=AP.vx_kd * ts_factor,
            sigma=AP.sigma,
            limit=AP.acc_x_sat_limit,
        )
        self.acc_y_from_vy = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.vy_kp,
            ki=AP.vy_ki / ts_factor,
            kd=AP.vy_kd * ts_factor,
            sigma=AP.sigma,
            limit=AP.acc_y_sat_limit,
        )
        self.acc_z_from_vz = PIDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.vz_kp,
            ki=AP.vz_ki / ts_factor,
            kd=AP.vz_kd * ts_factor,
            sigma=AP.sigma,
            limit=AP.acc_z_sat_limit,
        )

        # 角度环，从角度得到期望角速度
        self.roll_rate_from_roll = PDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.roll_kp,
            kd=AP.roll_kd * ts_factor,
            sigma=AP.sigma,
            limit=AP.roll_rate_sat_limit,
        )
        self.pitch_rate_from_pitch = PDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.pitch_kp,
            kd=AP.pitch_kd * ts_factor,
            sigma=AP.sigma,
            limit=AP.pitch_rate_sat_limit,
        )
        self.yaw_rate_from_yaw = PDControl(
            num_agent,
            dtype=dtype,
            Ts=ts_control,
            kp=AP.yaw_kp,
            kd=AP.yaw_kd * ts_factor,
            sigma=AP.sigma,
            limit=AP.yaw_rate_sat_limit,
        )

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

        thrust_per_motor = self.G_1_T_torch @ torch.cat((thrust_cmd, torque_x_cmd, torque_y_cmd, torque_z_cmd), 1)
        # force:N, torque:Nm

        # thrust_per_motor = ct * omega ** 2
        thrust_per_motor[thrust_per_motor < 0] = 0

        delta = torch.sqrt(thrust_per_motor / AP.k_t)

        return delta


if __name__ == "__main__":
    autopilot_func = Autopilot(0.02, 2000).to("cuda")
    autopilot = torch.jit.script(autopilot_func)

    print(1)
