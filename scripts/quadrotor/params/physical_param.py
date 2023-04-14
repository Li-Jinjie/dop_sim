import numpy as np
from ..tools.rotations import euler_2_quaternion_one

######################################################################################
#   Initial Conditions
######################################################################################
#   Initial conditions for MAV
east0 = 0.0  # initial east position
north0 = 0.0  # initial north position
up0 = 1.0  # initial up position
vx0 = 0.0  # initial velocity along body x-axis
vy0 = 0.0  # initial velocity along body y-axis
vz0 = 0.0  # initial velocity along body z-axis
phi0 = 0.0  # initial roll angle
theta0 = 0.0  # initial pitch angle
psi0 = 0.0  # initial yaw angle
wx0 = 0  # initial roll rate
wy0 = 0  # initial pitch rate
wz0 = 0  # initial yaw rate
# initial rotation speed of four motors, unit: RPM
o10 = 4000 / 1000  # kRPM
o20 = 4000 / 1000  # kRPM
o30 = 4000 / 1000  # kRPM
o40 = 4000 / 1000  # kRPM

Va0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
#   Quaternion State
q = euler_2_quaternion_one(phi0, theta0, psi0)
qw0 = q.item(0)
qx0 = q.item(1)
qy0 = q.item(2)
qz0 = q.item(3)

######################################################################################
#   Inertial Parameters, identified by me!
######################################################################################

# frame parameters
l_frame = 0.1372  # m
alpha_frame = 45.0 * np.pi / 180.0  # rad

# inertia parameters
mass = 1.4844  # kg    Add realsense and gps modules: 1.5344 kg; pure aircraft: 1.4844 kg, which is used in real-world experiments
gravity = 9.81  # m/s^2

Ixx = 0.0094  # kg m^2
Iyy = 0.0134  # kg m^2
Izz = 0.0145  # kg m^2
Ixz = 0.0

######################################################################################
#   Propeller thrust / torque parameters, identified by me!
######################################################################################
# change to kRPM to avoid quantization error
o_max = 24000 / 1000  # kRPM
o_min = 2600 / 1000  # kRPM
k_q = 3.7611e-10 * 1e6  # Nm/kRPM^2
k_t = 2.8158e-08 * 1e6  # N/kRPM^2

fc_max = 4.0 * k_t * (o_max**2)  # N

Tm = 0.0840  # time constant of motor

######################################################################################
#   aerodynamic parameters
######################################################################################
# aerodynamic drag model, refer to "A Comparative Study" Sun, et al.
kd_x = 0.26  # N s/m
kd_y = 0.28  # N s/m
kd_z = 0.42  # N s/m
k_h = 0.01  # N s^2/m^2

# downwash
dw_range_horiz = 1.5  # m
dw_range_vert = 4  # m
rp = 0.0775  # 3.05 inch = 0.07747 m
k_d1 = 4000
k_d2 = 0.65
k_d3 = -0.10


######################################################################################
#   Calculation Variables
######################################################################################
# from motor thrust to collective thrust and moments
l_s_alpha = l_frame * np.sin(alpha_frame)
l_c_alpha = l_frame * np.cos(alpha_frame)

# formulation: kt * [o1, o2, o3, o4]^T @ G_1 = [trust, tau_x, tau_y, tau_z]^T
G_1 = np.array(
    [
        [1, 1, 1, 1],
        [l_s_alpha, -l_s_alpha, -l_s_alpha, l_s_alpha],
        [-l_c_alpha, -l_c_alpha, l_c_alpha, l_c_alpha],
        [k_q / k_t, -k_q / k_t, k_q / k_t, -k_q / k_t],
    ]
)

#   gamma parameters pulled from page 36 (dynamics)
gamma = Ixx * Izz - (Ixz**2)
gamma1 = (Ixz * (Ixx - Iyy + Izz)) / gamma
gamma2 = (Izz * (Izz - Iyy) + (Ixz**2)) / gamma
gamma3 = Izz / gamma
gamma4 = Ixz / gamma
gamma5 = (Izz - Ixx) / Iyy
gamma6 = Ixz / Iyy
gamma7 = ((Ixx - Iyy) * Ixx + (Ixz**2)) / gamma
gamma8 = Ixx / gamma
