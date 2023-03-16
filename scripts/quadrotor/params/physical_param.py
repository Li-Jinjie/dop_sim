import numpy as np
from ..tools.rotations import euler_2_quaternion_one

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
o10 = 4000 / 1000  # kRPM
o20 = 4000 / 1000  # kRPM
o30 = 4000 / 1000  # kRPM
o40 = 4000 / 1000  # kRPM

Va0 = np.sqrt(vx0 ** 2 + vy0 ** 2 + vz0 ** 2)
#   Quaternion State
e = euler_2_quaternion_one(phi0, theta0, psi0)
ew0 = e.item(0)
ex0 = e.item(1)
ey0 = e.item(2)
ez0 = e.item(3)

######################################################################################
#   Physical Parameters, identified by me!
######################################################################################

# frame parameters
l_frame = 0.1372  # m
beta_frame = 45. * np.pi / 180.  # rad

# inertia parameters
mass = 1.5344  # kg
gravity = 9.81  # m/s^2

Ixx = 0.0094  # kg m^2
Iyy = 0.0134  # kg m^2
Izz = 0.0145  # kg m^2
Ixz = 0.

# aerodynamic drag model, refer to "A Comparative Study" Sun, et al.
kd_x = 0.26  # N s/m
kd_y = 0.26  # N s/m
kd_z = 0.42  # N s/m

k_h = 0.01  # N s^2/m^2

######################################################################################
#   Propeller thrust / torque parameters, identified by me!
######################################################################################
# change to kRPM to avoid quantization error
o_max = 24000 / 1000  # kRPM
o_min = 2600 / 1000  # kRPM
k_q = 3.7611e-010 * 1e6  # Nm/kRPM^2
k_t = 2.8158e-08 * 1e6  # N/kRPM^2

Tm = 0.0760  # time constant of motor  # 这个参数参考《多旋翼飞行器设计与控制》 全权

collective_f_max = 4.0 * k_t * o_max ** 2  # N

######################################################################################
#   Calculation Variables
######################################################################################
# from motor thrust to collective thrust and moments
l_s_beta = l_frame * np.sin(beta_frame)
l_c_beta = l_frame * np.cos(beta_frame)

# formulation: kt * [o1, o2, o3, o4]^T @ G_1 = [trust, tau_x, tau_y, tau_z]^T
G_1 = np.array([[1, 1, 1, 1],
                [l_s_beta, -l_s_beta, -l_s_beta, l_s_beta],
                [l_c_beta, l_c_beta, -l_c_beta, -l_c_beta],
                [-k_q / k_t, k_q / k_t, -k_q / k_t, k_q / k_t]])

#   gamma parameters pulled from page 36 (dynamics)
gamma = Ixx * Izz - (Ixz ** 2)
gamma1 = (Ixz * (Ixx - Iyy + Izz)) / gamma
gamma2 = (Izz * (Izz - Iyy) + (Ixz ** 2)) / gamma
gamma3 = Izz / gamma
gamma4 = Ixz / gamma
gamma5 = (Izz - Ixx) / Iyy
gamma6 = Ixz / Iyy
gamma7 = ((Ixx - Iyy) * Ixx + (Ixz ** 2)) / gamma
gamma8 = Ixx / gamma