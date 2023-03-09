import sys

import numpy as np
from models.tools.rotations import euler_2_quaternion_one

######################################################################################
#   Initial Conditions
######################################################################################
#   Initial conditions for MAV
north0 = 0.  # initial north position
east0 = 0.  # initial east position
down0 = -1.0  # initial down position
vx0 = 0.  # initial velocity along body x-axis
vy0 = 0.  # initial velocity along body y-axis
vz0 = 0.  # initial velocity along body z-axis
phi0 = 0.  # initial roll angle
theta0 = 0.  # initial pitch angle
psi0 = 0.  # initial yaw angle
p0 = 0  # initial roll rate
q0 = 0  # initial pitch rate
r0 = 0  # initial yaw rate
# initial rotation speed of four motors, unit: RPM
o10 = 4000
o20 = 4000
o30 = 4000
o40 = 4000

Va0 = np.sqrt(vx0 ** 2 + vy0 ** 2 + vz0 ** 2)
#   Quaternion State
e = euler_2_quaternion_one(phi0, theta0, psi0)
ew0 = e.item(0)
ex0 = e.item(1)
ey0 = e.item(2)
ez0 = e.item(3)

######################################################################################
#   Physical Parameters
######################################################################################
# 参考《多旋翼飞行器设计与控制》 全权

mass = 1.0230  # kg
Jx = 0.0095  # kg m^2
Jy = 0.0095  # kg m^2
Jz = 0.0186  # kg m^2
Jxz = 0.

l_frame = 0.2223  # m
beta_frame = 45. * np.pi / 180.  # rad

gravity = 9.81  # m/s^2

# aerodynamic drag model, refer to "A Comparative Study" Sun, et al.
kd_x = 0.26  # N s/m
kd_y = 0.26  # N s/m
kd_z = 0.42  # N s/m

k_h = 0.01  # N s^2/m^2

######################################################################################
#   Propeller thrust / torque parameters
######################################################################################
o_max = 12000  # RPM
o_min = 2000  # RPM

c_q = 2.9250e-09  # Nm/RPM^2
c_t = 1.4865e-07  # N/RPM^2

Tm = 0.0760  # time constant of motor

######################################################################################
#   Calculation Variables
######################################################################################
# from motor thrust to collective thrust and moments
l_s_beta = l_frame * np.sin(beta_frame)
l_c_beta = l_frame * np.cos(beta_frame)
G_1 = c_t * np.array([[1, 1, 1, 1],
                      [l_s_beta, -l_s_beta, -l_s_beta, l_s_beta],
                      [l_c_beta, l_c_beta, -l_c_beta, -l_c_beta],
                      [-c_q / c_t, c_q / c_t, -c_q / c_t, c_q / c_t]])

#   gamma parameters pulled from page 36 (dynamics)
gamma = Jx * Jz - (Jxz ** 2)
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + (Jxz ** 2)) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + (Jxz ** 2)) / gamma
gamma8 = Jx / gamma
