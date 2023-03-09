#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: qd_dynamics.py
Date: 11/2/2022 7:17 PM
LastEditors: LI Jinjie
LastEditTime: 11/2/2022 7:17 PM
Description: file content
"""

import torch
import torch.nn as nn
import math

from potato_msgs.central.msg_mul_state import MsgMulState
from potato_msgs.central.quadrotor.msg_mul_delta import MsgMulDelta

from ..parameters import aerosonde_parameters as QMAV
from ...tools.rotations import quaternion_2_rotation, quaternion_2_euler
from .rigid_body_use_vw import RigidBodyUseVw
from ...tools.ode import ODE
from ...tools.saturate import saturate_w_float_limit


def _update_wind_related_states(state: torch.Tensor, R: torch.Tensor):
    V_wind_i = state[:, 28:31, :]

    # R: body to inertial  R.T: inertial to body
    V_wind_b = R.transpose(1, 2) @ V_wind_i  # to the body frame

    # velocity vector relative to the airmass
    u_r = state[:, 16:17, :] - V_wind_b[:, 0:1, :]
    v_r = state[:, 17:18, :] - V_wind_b[:, 1:2, :]
    w_r = state[:, 18:19, :] - V_wind_b[:, 2:3, :]

    # compute airspeed
    Va = torch.sqrt(u_r**2 + v_r**2 + w_r**2)
    # compute angle of attack
    alpha = (u_r == 0) * math.pi / 2 + (u_r != 0) * torch.atan2(w_r, u_r)
    # compute sideslip angle
    beta = (Va == 0) * 0 + (Va != 0) * torch.asin(v_r / Va)

    state[:, 22:23, :] = Va
    state[:, 24:25, :] = alpha
    state[:, 25:26, :] = beta

    return state


def _update_the_other_states(state: torch.Tensor, R: torch.Tensor):
    # update the class structure for the true state:
    phi, theta, psi = quaternion_2_euler(state[:, 9:13, :])
    pdot = R @ state[:, 16:19, :]  # R: body to inertial

    state[:, 6:7, :] = phi
    state[:, 7:8, :] = theta
    state[:, 8:9, :] = psi

    state[:, 23:24, :] = torch.unsqueeze(torch.unsqueeze(torch.norm(pdot, dim=[1, 2]), 1), 2)  # Vg
    state[:, 26:27, :] = torch.asin(pdot[:, 2:3, :] / state[:, 23:24, :])  # gamma
    state[:, 27:28, :] = torch.atan2(pdot[:, 1:2, :], pdot[:, 0:1, :])  # chi

    return state


class QdDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_1_torch = nn.Parameter(torch.tensor(QMAV.G_1), False)
        self.ode_rigid_body = ODE(RigidBodyUseVw)

    def forward(self, state: torch.Tensor, delta: torch.Tensor, dt: float):
        """
        Integrate the differential equations defining a_dynamics, update sensors
        delta = (aileron, elevator, throttle, rudder) are the control inputs
        Ts is the time step between function calls.

        delta = [aileron, elevator, throttle, rudder]
        state = [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
        p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd, xxx]
        input = [fx, fy, fz, l, m, n]
        """
        # get forces and moments acting on rigid body
        state, force_w_moment_b, R = self._update_forces_moments_and_wind_related_state(state, delta)

        state = self._update_dynamics_related_states(state, force_w_moment_b, dt)

        state = _update_the_other_states(state, R)

        return state

    def _motor_thrust_torque(self, delta: torch.Tensor) -> torch.Tensor:
        # delta = [o1, o2, o3, o4]
        # thrust_torque = [thrust, torque_x, torque_y, torque_z]

        # 对delta进行限幅
        delta = saturate_w_float_limit(delta, low_limit=float(QMAV.o_min), up_limit=float(QMAV.o_max))

        o1 = delta[:, 0:1, :]
        o2 = delta[:, 1:2, :]
        o3 = delta[:, 2:3, :]
        o4 = delta[:, 3:4, :]

        # ox的单位可以是转速, rad/s；也可以是RPM/s，只要与电机螺旋桨的拉力常数和扭矩常数单位对应上就可以。
        thrust_torque = self.G_1_torch @ torch.cat((o1**2, o2**2, o3**2, o4**2), 1)

        return thrust_torque

    def _update_forces_moments_and_wind_related_state(self, state: torch.Tensor, delta: torch.Tensor):
        """calculate forces and moments
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta = [o1, o2, o3, o4]
        state = [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
        p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, xxx]
        :return: Forces and Moments on the UAV [fx, fy, fz, l, m, n]
        """

        # compute propeller thrust and torque
        thrust_torque = self._motor_thrust_torque(delta)

        T = thrust_torque[:, 0:1, :]
        Mx = thrust_torque[:, 1:2, :]
        My = thrust_torque[:, 2:3, :]
        Mz = thrust_torque[:, 3:4, :]

        # ------- update_wind_related_states --------
        R = quaternion_2_rotation(state[:, 9:13, :])  # R: body to inertial  R.T: inertial to body

        V_wind_i = state[:, 28:31, :]

        # R: body to inertial  R.T: inertial to body
        V_wind_b = R.transpose(1, 2) @ V_wind_i  # to the body frame

        # velocity vector relative to the airmass
        u_r = state[:, 16:17, :] - V_wind_b[:, 0:1, :]
        v_r = state[:, 17:18, :] - V_wind_b[:, 1:2, :]
        w_r = state[:, 18:19, :] - V_wind_b[:, 2:3, :]

        # compute airspeed
        Va = torch.sqrt(u_r**2 + v_r**2 + w_r**2)
        # compute angle of attack
        alpha = (u_r == 0) * math.pi / 2 + (u_r != 0) * torch.atan2(w_r, u_r)
        # compute sideslip angle
        beta = (Va == 0) * 0 + (Va != 0) * torch.asin(v_r / Va)

        state[:, 22:23, :] = Va
        state[:, 24:25, :] = alpha
        state[:, 25:26, :] = beta

        # compute force in body frame: thrust + aerodynamic drag force

        fx = 0 + -QMAV.kd_x * u_r
        fy = 0 + -QMAV.kd_y * v_r
        fz = -T + -QMAV.kd_z * w_r + QMAV.k_h * (u_r**2 + v_r**2)

        # transform forces from body frame to inertial frame
        # R: body to inertial  R.T: inertial to body
        f_i = R @ torch.cat((fx, fy, fz), 1)

        return state, torch.cat([f_i, Mx, My, Mz], dim=1), R

    def _update_dynamics_related_states(self, state: torch.Tensor, force_w_moment_b: torch.Tensor, dt: float):
        # ode_state = [north, east, down, vx, vy, vz, ew, ex, ey, ez, p, q, r]
        ode_state = torch.cat((state[:, 3:6, :], state[:, 13:16, :], state[:, 9:13, :], state[:, 19:22]), 1)

        ode_state_new = self.ode_rigid_body(ode_state, force_w_moment_b, dt)

        # normalize the quaternion
        ew = ode_state_new[:, 6:7, :]
        ex = ode_state_new[:, 7:8, :]
        ey = ode_state_new[:, 8:9, :]
        ez = ode_state_new[:, 9:10, :]
        normE = torch.sqrt(ew**2.0 + ex**2.0 + ey**2.0 + ez**2.0)
        ode_state_new[:, 6:10, :] = ode_state_new[:, 6:10, :] / normE

        # put the ode_state into state
        state[:, 3:6, :] = ode_state_new[:, 0:3, :]  # n e d
        state[:, 13:16, :] = ode_state_new[:, 3:6, :]  # vx, vy, vz
        state[:, 9:13, :] = ode_state_new[:, 6:10, :]  # e
        state[:, 19:22] = ode_state_new[:, 10:13, :]  # p q r

        return state


if __name__ == "__main__":
    # from rigid_body_use_vw import RigidBodyUseVw
    #
    # ode_func = ODE(RigidBodyUseVw).to("cuda")

    fw_dynamics_func = QdDynamics().to("cuda")
    fw_dynamics = torch.jit.script(fw_dynamics_func)
    # fw_dynamics = fw_dynamics_func

    rows_index = [10000, 10001, 10002, 10003]
    dt = 0.02

    state = MsgMulState()
    state.set_zero(rows_index)

    delta = MsgMulDelta()
    delta.set_zero(rows_index)

    new_state_tensor = fw_dynamics(state.to_tensor(), delta.to_tensor(), dt)

    state.from_tensor(rows_index, new_state_tensor)

    print(1)
