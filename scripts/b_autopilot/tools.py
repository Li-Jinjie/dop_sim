#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: tools.py
Date: 2023/3/1 上午10:09
Description: 
"""


import torch
import math


def wrap(chi_1: torch.Tensor, chi_2: torch.Tensor, limit: float = math.pi):
    """
    wrap chi_1, so that it is within +-pi of chi_2
    """
    # pay attention to the while logic in parallel computation
    more_pi_flag = (chi_1 - chi_2) > limit
    while torch.sum(more_pi_flag).item() != 0:
        chi_1 = more_pi_flag * (chi_1 - 2.0 * limit) + ~more_pi_flag * chi_1
        more_pi_flag = (chi_1 - chi_2) > limit

    less_m_pi_flag = (chi_1 - chi_2) < -limit
    while torch.sum(less_m_pi_flag).item() != 0:
        chi_1 = less_m_pi_flag * (chi_1 + 2.0 * limit) + ~less_m_pi_flag * chi_1
        less_m_pi_flag = (chi_1 - chi_2) < -limit
    return chi_1


def saturate_w_tensor_limit(input: torch.Tensor, low_limit: torch.Tensor, up_limit: torch.Tensor) -> torch.Tensor:
    output = (
        (input <= low_limit) * low_limit
        + (input >= up_limit) * up_limit
        + (~(input <= low_limit) * ~(input >= up_limit)) * input
    )

    return output


def saturate_w_float_limit(input: torch.Tensor, low_limit: float, up_limit: float) -> torch.Tensor:
    output = (
        (input <= low_limit) * low_limit
        + (input >= up_limit) * up_limit
        + (~(input <= low_limit) * ~(input >= up_limit)) * input
    )

    return output
