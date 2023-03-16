"""
wrap chi_1, so that it is within +-pi of chi_2
"""
import torch
import math


def wrap(chi_1: torch.Tensor, chi_2: torch.Tensor, limit: float = math.pi):
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
