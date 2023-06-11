import sys

import numpy as np
from . import physical_param as MAV

mass = MAV.mass
gravity = MAV.gravity  # gravity constant
sigma = 0.05  # low pass filter gain for derivative in PID

fc_max = MAV.fc_max

alpha = MAV.alpha_frame
l_frame = MAV.l_frame

k_t = MAV.k_t
k_q = MAV.k_q

# throttle to thrust
k_th = 17.666
b_th = -1.206

# duration
t_all = 705.0  # s

# ----------position loop-------------
"""
参数是在Ts=0.02s的条件下调试的
"""
# pn_kp = 0.5
# # pn_ki = 0.00
# pn_kd = 0.001
# vx_sat_limit = 10  # m/s
#
# pe_kp = 0.5
# # pe_ki = 0.00
# pe_kd = 0.001
# vy_sat_limit = 10  # m/s
#
# pd_kp = 2.5
# # pd_ki = 0.00
# pd_kd = 0.00
# vz_sat_limit = 10  # m/s

# # ---------- velocity loop -------------
# # the outputs are phi_desired and theta_desired
# vh_kp = 2.4  # horizontal
# vh_ki = 0.5
# vh_kd = 0.1
#
# vx_kp = vh_kp
# vx_ki = vh_ki
# vx_kd = vh_kd
# acc_x_sat_limit = 2.0 * gravity
#
# vy_kp = vh_kp
# vy_ki = vh_ki
# vy_kd = vh_kd
# acc_y_sat_limit = 2.0 * gravity
#
# # the output is collective thrust
# vz_kp = 8.0  # 8.0
# vz_ki = 2.0  # 2.0
# vz_kd = 0.1
# acc_z_sat_limit = 2.0 * gravity

# # ---------- attitude loop -------------
# angle_kp = 8.0
# angle_kd = 0.1
# angle_input_limit = 30.0 * np.pi / 180.0  # rad
# angle_rate_sat_limit = 30.0 * np.pi / 180.0  # rad/s
#
# roll_kp = angle_kp
# # roll_ki = 0.0
# roll_kd = angle_kd
# roll_input_limit = angle_input_limit
# roll_rate_sat_limit = angle_rate_sat_limit
#
# pitch_kp = angle_kp
# # pitch_ki = 0.0
# pitch_kd = angle_kd
# pitch_input_limit = angle_input_limit
# pitch_rate_sat_limit = angle_rate_sat_limit
#
# yaw_kp = 7.0
# # yaw_ki = 0.0
# yaw_kd = 0.2
# yaw_rate_sat_limit = 30.0 * np.pi / 180.0  # rad/s

# ---------- attitude rate loop -------------
roll_rate_kp = 0.3
roll_rate_ki = 0.01
roll_rate_kd = 0.005

pitch_rate_kp = 0.3
pitch_rate_ki = 0.01
pitch_rate_kd = 0.005

yaw_rate_kp = 0.13
yaw_rate_ki = 0.01
yaw_rate_kd = 0.005

# ---------- power distribution -------------
# formulation: kt * [o1, o2, o3, o4]^T = G_1_inv * [trust, tau_x, tau_y, tau_z]^T
# G_1_inv = np.linalg.inv(MAV.G_1)  # numerical inverse

csc_alpha_div_4_l_frame = 1 / (4 * l_frame * np.sin(alpha))
sec_alpha_div_4_l_frame = 1 / (4 * l_frame * np.cos(alpha))

G_1_inv = np.array(
    [
        [1 / 4, -csc_alpha_div_4_l_frame, -sec_alpha_div_4_l_frame, -k_t / (4 * k_q)],
        [1 / 4, csc_alpha_div_4_l_frame, sec_alpha_div_4_l_frame, -k_t / (4 * k_q)],
        [1 / 4, csc_alpha_div_4_l_frame, -sec_alpha_div_4_l_frame, k_t / (4 * k_q)],
        [1 / 4, -csc_alpha_div_4_l_frame, sec_alpha_div_4_l_frame, k_t / (4 * k_q)],
    ]
)
