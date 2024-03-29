#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: mul_quadrotors.py
Date: 11/5/2022 10:32 AM
LastEditors: LI Jinjie
LastEditTime: 11/5/2022 10:32 AM
Description: file content
"""
import torch.nn as nn
import torch

# ===== quadrotor related =====
from .a_dynamics import QdDynamics
from .b_autopilot import AtpRate


class MulQuadrotors(nn.Module):
    def __init__(
        self,
        num_agent: int,
        ts_sim: float,
        ts_control: float,
        dtype=torch.float64,
        has_downwash=True,
        has_motor_model=True,
        has_battery=True,
    ):
        super().__init__()
        self.autopilot = AtpRate(num_agent, ts_control, has_battery, dtype)
        self.dynamics = QdDynamics(ts_sim, has_downwash, has_motor_model)

        self.ts_ctl = ts_control
        self.ctl_t = 999.0  # counting time for control update
        self.delta = nn.Parameter(torch.zeros([num_agent, 4, 1], dtype=dtype), False)

        self.all_sim_t = 0.0

    def forward(self, ts_sim: float, ego_states: torch.Tensor, body_rate_cmd: torch.Tensor):
        if self.ctl_t > self.ts_ctl:
            self.delta = self.autopilot(ego_states, body_rate_cmd, self.all_sim_t)
            self.ctl_t = 0.0

        ego_states_new = self.dynamics(ts_sim, ego_states, self.delta)

        self.ctl_t += ts_sim
        self.all_sim_t += ts_sim

        return ego_states_new
