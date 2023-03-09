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
from b_autopilot import Autopilot


class MulQuadrotors:
    def __init__(self, num_agent: int, ego_states: MsgMulState, sim_ts: float, control_ts: float, send_to):
        """class of one fixed-wing airplane.

        Args:
            num_agent: the total number of agents
            sim_ts: time step for simulation
            control_ts: time step for control
            send_to: method for communication with viewer or algo ends
        """
        # initiate basic attributes for swarm

        # DOP
        dop_computation_func = DOPComputation(control_ts, num_agent).requires_grad_(False)

        if torch.cuda.is_available():
            dop_computation_func = dop_computation_func.to("cuda")  # data-oriented

        self.dop_computation = torch.jit.script(dop_computation_func)  # 起飞！
        # self.dop_computation = dop_computation_func  # debug only

    # @timer
    def update(self, ego_states: MsgMulState):

        value_tensor = self.dop_computation(ego_states.to_tensor(), self.path_cmd.to_tensor(), self.sim_ts)

        return value_tensor


class DOPComputation(nn.Module):
    def __init__(self, ts_control: float, num_agent: int, dtype=torch.float64):
        super().__init__()

        self.autopilot = Autopilot(num_agent, ts_control, dtype)
        self.dynamics = QdDynamics()

    def forward(self, ego_states: torch.Tensor, body_rate_cmd: torch.Tensor, sim_ts: float):

        delta_cmd = self.autopilot(ego_states, body_rate_cmd)
        ego_states_new = self.dynamics(ego_states, delta_cmd, sim_ts)

        return ego_states_new
