#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: rigid_body_use_vw.py
Date: 10/15/2022 7:54 PM
LastEditors: LI Jinjie
LastEditTime: 10/15/2022 7:54 PM
Description: file content
"""

import torch
import torch.nn as nn

from ..params import physical_param as QMAV


class RigidBodyUseVw(nn.Module):
    def __init__(self):
        super().__init__()
        self.mass = nn.Parameter(torch.tensor(QMAV.mass), False)
        self.Jy = nn.Parameter(torch.tensor(QMAV.Iyy), False)
        self.gamma1 = nn.Parameter(torch.tensor(QMAV.gamma1), False)
        self.gamma2 = nn.Parameter(torch.tensor(QMAV.gamma2), False)
        self.gamma3 = nn.Parameter(torch.tensor(QMAV.gamma3), False)
        self.gamma4 = nn.Parameter(torch.tensor(QMAV.gamma4), False)
        self.gamma5 = nn.Parameter(torch.tensor(QMAV.gamma5), False)
        self.gamma6 = nn.Parameter(torch.tensor(QMAV.gamma6), False)
        self.gamma7 = nn.Parameter(torch.tensor(QMAV.gamma7), False)
        self.gamma8 = nn.Parameter(torch.tensor(QMAV.gamma8), False)

    def forward(self, state: torch.Tensor, force_w_torque_b: torch.Tensor):
        """
        for the a_dynamics xdot = f(x, u), returns f(x, u)
        state = [east, north, up, vx, vy, vz, ew, ex, ey, ez, p, q, r]
        input = [fx, fy, fz, l, m, n]
        """

        # extract the states
        # east = state[:, 0:1, :]
        # north = state[:, 1:2, :]
        # up = state[:, 2:3, :]

        # vx, vy, vz, ew, ex, ey, ez, p, q, r = fork_tensor(state, [16, 17, 18, 9, 10, 11, 12, 19, 20, 21])
        # 注，这里必须写成这种固定长度的才能够jit加速

        vx = state[:, 3:4, :]
        vy = state[:, 4:5, :]
        vz = state[:, 5:6, :]

        ew = state[:, 6:7, :]
        ex = state[:, 7:8, :]
        ey = state[:, 8:9, :]
        ez = state[:, 9:10, :]

        p = state[:, 10:11, :]
        q = state[:, 11:12, :]
        r = state[:, 12:13, :]

        #   extract forces/moments
        fx_i = force_w_torque_b[:, 0:1, :]
        fy_i = force_w_torque_b[:, 1:2, :]
        fz_i = force_w_torque_b[:, 2:3, :]
        l = force_w_torque_b[:, 3:4, :]
        m = force_w_torque_b[:, 4:5, :]
        n = force_w_torque_b[:, 5:6, :]

        # position kinematics, in the inertial coordination
        east_dot = vx
        north_dot = vy
        up_dot = vz

        # position dynamics, velocity, f_i is in inertial coordination
        vx_dot = fx_i / QMAV.mass
        vy_dot = fy_i / QMAV.mass
        vz_dot = fz_i / QMAV.mass - QMAV.gravity  # ENU

        # rotational kinematics
        ew_dot = (0 - p * ex - q * ey - r * ez) / 2
        ex_dot = (p * ew + 0 + r * ey - q * ez) / 2
        ey_dot = (q * ew - r * ex + 0 + p * ez) / 2
        ez_dot = (r * ew + q * ex - p * ey + 0) / 2

        # rotational dynamics, (l,m,n) is in body coordination
        p_dot = QMAV.gamma1 * p * q - QMAV.gamma2 * q * r + QMAV.gamma3 * l + QMAV.gamma4 * n
        q_dot = QMAV.gamma5 * p * r - QMAV.gamma6 * (p**2 - r**2) + m / QMAV.Iyy
        r_dot = QMAV.gamma7 * p * q - QMAV.gamma1 * q * r + QMAV.gamma4 * l + QMAV.gamma8 * n

        # collect the derivative of the states
        state_dot = torch.cat(
            (
                east_dot,
                north_dot,
                up_dot,
                vx_dot,
                vy_dot,
                vz_dot,
                ew_dot,
                ex_dot,
                ey_dot,
                ez_dot,
                p_dot,
                q_dot,
                r_dot,
            ),
            1,
        )
        return state_dot


if __name__ == "__main__":

    rigid_body_func = RigidBodyUseVw().to("cuda")
    rigid_body = torch.jit.script(rigid_body_func)

    print(1)
