#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: ode.py
Date: 10/15/2022 8:21 PM
LastEditors: LI Jinjie
LastEditTime: 10/15/2022 8:21 PM
Description: file content
"""
import numpy as np
import torch
import torch.nn as nn


class ODE(nn.Module):
    def __init__(self, Func):
        # Func必须是Module
        super().__init__()
        self.ode_method = "RK4"
        self.func = Func()

    def forward(self, x: torch.Tensor, u: torch.Tensor, dt: float):
        # 后期来说，dt可能是可变的，所以放在这里作为参数传入
        if self.ode_method == "RK4":
            k1 = self.func(x, u)
            k2 = self.func(x + dt / 2.0 * k1, u)
            k3 = self.func(x + dt / 2.0 * k2, u)
            k4 = self.func(x + dt * k3, u)
            x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x


if __name__ == "__main__":
    from models.fixed_wing.a_dynamics.rigid_body_use_vw import RigidBodyUseVw

    ode_func = ODE(RigidBodyUseVw).to("cuda")
    ode = torch.jit.script(ode_func)

    state = torch.zeros([2000, 13, 1]).to("cuda")
    input = torch.ones([2000, 6, 1]).to("cuda")

    state = ode(state, input, 0.02)

    print(1)
