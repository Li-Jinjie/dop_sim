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
import time
import torch.nn as nn
import torch

# ===== quadrotor related =====
from a_dynamics import QdDynamics
from b_autopilot import AtpRate


class MulQd(nn.Module):
    def __init__(self, num_agent: int, ts_control: float, dtype=torch.float64):
        super().__init__()
        self.autopilot = AtpRate(num_agent, ts_control, dtype)
        self.dynamics = QdDynamics()

    def forward(self, ego_states: torch.Tensor, body_rate_cmd: torch.Tensor, sim_ts: float):
        delta_cmd = self.autopilot(ego_states, body_rate_cmd)
        ego_states_new = self.dynamics(ego_states, delta_cmd, sim_ts)

        return ego_states_new


if __name__ == "__main__":
    mul_qd = MulQd(2, 0.02, torch.float64)
