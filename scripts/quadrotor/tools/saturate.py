#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: saturate.py
Date: 10/20/2022 10:10 AM
LastEditors: LI Jinjie
LastEditTime: 10/20/2022 10:10 AM
Description: file content
"""
import torch


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
