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

from ..params import physical_param as QMAV
from ..tools.rotations import quaternion_2_rotation, quaternion_2_euler
from .rigid_body_use_vw import RigidBodyUseVw
from ..tools.ode import ODE
from ..tools.saturate import saturate_w_float_limit


def _update_wind_related_states(state: torch.Tensor, r_mtx: torch.Tensor):
    V_wind_i = state[:, 28:31, :]

    # R: body to inertial  R.T: inertial to body
    V_wind_b = r_mtx.transpose(1, 2) @ V_wind_i  # to the body frame

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


def _update_other_states(state: torch.Tensor, r_mtx: torch.Tensor):
    # update the class structure for the true state:
    phi, theta, psi = quaternion_2_euler(state[:, 9:13, :])
    pdot = r_mtx @ state[:, 16:19, :]  # R: body to inertial

    state[:, 6:7, :] = phi
    state[:, 7:8, :] = theta
    state[:, 8:9, :] = psi

    state[:, 23:24, :] = torch.unsqueeze(torch.unsqueeze(torch.norm(pdot, dim=[1, 2]), 1), 2)  # Vg
    state[:, 26:27, :] = torch.asin(pdot[:, 2:3, :] / state[:, 23:24, :])  # gamma
    state[:, 27:28, :] = torch.atan2(pdot[:, 1:2, :], pdot[:, 0:1, :])  # chi

    return state


class QdDynamics(nn.Module):
    def __init__(self, ts_sim, has_downwash, has_motor_model):
        super().__init__()
        self.G_1_torch = nn.Parameter(torch.tensor(QMAV.G_1), False)
        self.ode_rigid_body = ODE(RigidBodyUseVw)

        self.has_downwash = has_downwash
        self.has_motor_model = has_motor_model

        self.motor_alpha = math.exp(-ts_sim / QMAV.Tm)

    def forward(self, dt: float, state: torch.Tensor, delta: torch.Tensor):
        """
        Integrate the differential equations defining a_dynamics, update sensors
        delta = (aileron, elevator, throttle, rudder) are the control inputs
        Ts is the time step between function calls.

        delta = [o1, o2, o3, o4]
        state = [player, group, HP, east, north, up, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
        p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd, o1, o2, o3, o4]
        input = [fx, fy, fz, l, m, n]
        """
        if self.has_motor_model:
            delta = self._motor_model(state, delta)

        # get forces and torques acting on rigid body
        state, forces_i_torques_b, r_mtx = self._update_forces_torques(state, delta)

        state = self._update_dynamics(state, forces_i_torques_b, dt)

        state = _update_other_states(state, r_mtx)

        state[:, 31:35, :] = delta[:, 0:4, :]

        return state

    def _update_forces_torques(self, state: torch.Tensor, delta: torch.Tensor):
        """calculate forces and torques
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta = [o1, o2, o3, o4]
        state = [player, group, HP, east, north, up, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
        p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, xxx]
        :return: Forces and Torques on the UAV [fx, fy, fz, l, m, n]
        """

        # compute propeller thrust and torque
        thrust_torque = self._rotor_model(delta)

        f_c = thrust_torque[:, 0:1, :]

        fu_x = 0
        fu_y = 0
        fu_z = f_c

        tau_x = thrust_torque[:, 1:2, :]
        tau_y = thrust_torque[:, 2:3, :]
        tau_z = thrust_torque[:, 3:4, :]

        # ------- update_wind_related_states --------
        r_mtx = quaternion_2_rotation(state[:, 9:13, :])  # r_mtx: body to inertial  r_mtx.T: inertial to body

        V_wind_i = state[:, 28:31, :]

        # r_mtx: body to inertial  r_mtx.T: inertial to body
        V_wind_b = r_mtx.transpose(1, 2) @ V_wind_i  # to the body frame

        # velocity vector relative to the airmass
        u_r = state[:, 16:17, :] - V_wind_b[:, 0:1, :]
        v_r = state[:, 17:18, :] - V_wind_b[:, 1:2, :]
        w_r = state[:, 18:19, :] - V_wind_b[:, 2:3, :]

        # # TODO: check these values, from NED to ENU
        # compute airspeed
        Va = torch.sqrt(u_r**2 + v_r**2 + w_r**2)
        # compute angle of attack
        alpha = (u_r == 0) * math.pi / 2 + (u_r != 0) * torch.atan2(w_r, u_r)
        # compute sideslip angle
        beta = (Va == 0) * 0 + (Va != 0) * torch.asin(v_r / Va)

        state[:, 22:23, :] = Va
        state[:, 24:25, :] = alpha
        state[:, 25:26, :] = beta

        # ------- compute air drag forces --------
        fa_x = -QMAV.kd_x * u_r
        fa_y = -QMAV.kd_y * v_r
        fa_z = -QMAV.kd_z * w_r + QMAV.k_h * (u_r**2 + v_r**2)

        # ------- compute force in body frame: thrust + aerodynamic drag force --------
        fx_b = fu_x + fa_x
        fy_b = fu_y + fa_y
        fz_b = fu_z + fa_z

        # transform forces from body frame to inertial frame
        # r_mtx: body to inertial  r_mtx.T: inertial to body
        f_i = r_mtx @ torch.cat((fx_b, fy_b, fz_b), 1)

        if self.has_downwash:
            # ------- compute downwash forces --------
            east = state[:, 3:4, :]
            north = state[:, 4:5, :]
            up = state[:, 5:6, :]

            # others - ego. see column vector
            # leverage PyTorch broadcasting to accelerate the computation
            dx_bc = torch.transpose(east, 0, 1) - east  # bc refers to broadcast
            dy_bc = torch.transpose(north, 0, 1) - north
            dz_bc = torch.transpose(up, 0, 1) - up

            # avoid calculating too large numbers, unnecessary
            dx_bc[dx_bc > QMAV.dw_range_horiz] = QMAV.dw_range_horiz
            dy_bc[dy_bc > QMAV.dw_range_horiz] = QMAV.dw_range_horiz
            dist_horiz = torch.sqrt(dx_bc**2 + dy_bc**2)

            is_close = dist_horiz < QMAV.dw_range_horiz
            is_lower = (dz_bc > 0) * (dz_bc < QMAV.dw_range_vert)
            is_valid = is_close * is_lower

            dist_horiz = dist_horiz * is_valid
            dz_bc = dz_bc * is_valid

            # dz_bc is forbidden to be zero
            dz_zero_flag = dz_bc == 0
            dz_bc[dz_zero_flag] = 1

            fdz_bc = (
                -QMAV.k_d1
                * (QMAV.rp / 4 / dz_bc) ** 2
                * torch.exp(-0.5 * (dist_horiz / (QMAV.k_d2 * dz_bc + QMAV.k_d3)) ** 2)
            )
            fdz_bc[dz_zero_flag] = 0

            fd_z = torch.sum(fdz_bc, 1).unsqueeze(1)

            f_i[:, 2:3, :] += fd_z

        return state, torch.cat([f_i, tau_x, tau_y, tau_z], dim=1), r_mtx

    def _update_dynamics(self, state: torch.Tensor, forces_i_torques_b: torch.Tensor, dt: float):
        # ode_state = [east, north, up, vx, vy, vz, ew, ex, ey, ez, p, q, r]
        ode_state = torch.cat((state[:, 3:6, :], state[:, 13:16, :], state[:, 9:13, :], state[:, 19:22]), 1)

        ode_state_new = self.ode_rigid_body(ode_state, forces_i_torques_b, dt)

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

    def _motor_model(self, state: torch.Tensor, delta_cmd: torch.Tensor) -> torch.Tensor:
        # first-order system, which can be represented as a low-pass filter
        delta_real = state[:, 31:35, :]
        delta_real = self.motor_alpha * delta_real + (1 - self.motor_alpha) * delta_cmd
        return delta_real

    def _rotor_model(self, delta: torch.Tensor) -> torch.Tensor:
        """quadratic rotor model"""
        # delta = [o1, o2, o3, o4]
        # thrust_torque = [thrust, torque_x, torque_y, torque_z]

        # 对delta进行限幅
        delta = saturate_w_float_limit(delta, low_limit=float(QMAV.o_min), up_limit=float(QMAV.o_max))

        # ox的单位可以是转速, rad/s；也可以是RPM，只要与电机螺旋桨的拉力常数和扭矩常数单位对应上就可以。这里是kRPM
        o1 = delta[:, 0:1, :]
        o2 = delta[:, 1:2, :]
        o3 = delta[:, 2:3, :]
        o4 = delta[:, 3:4, :]

        thrust_per_rotor = QMAV.k_t * torch.cat((o1**2, o2**2, o3**2, o4**2), 1)

        thrust_torque = self.G_1_torch @ thrust_per_rotor

        return thrust_torque


# # # Test Code
# if __name__ == "__main__":
#     # from rigid_body_use_vw import RigidBodyUseVw
#     #
#     # ode_func = ODE(RigidBodyUseVw).to("cuda")
#
#     fw_dynamics_func = QdDynamics().to("cuda")
#     fw_dynamics = torch.jit.script(fw_dynamics_func)
#     # fw_dynamics = fw_dynamics_func
#
#     rows_index = [10000, 10001, 10002, 10003]
#     dt = 0.02
#
#     new_state_tensor = fw_dynamics(state_tensor, delta_tensor, dt)
#
#     print(1)
