"""
various tools to be used in mavPySim
"""
import numpy as np
import torch


def quaternion_2_euler(quaternion):
    """
    Conversion from quaternions to euler angles, page 259
    Args:
        num_airplane: the total number of airplanes
        device: the device that the computation runs on.
        dtype: the type of the data
        quaternion: an 4*num_airplane tensor, quaternion of each airplane

    Returns: eular angle
        phi: an 1*num_airplane tensor
        theta: an 1*num_airplane tensor
        psi: an 1*num_airplane tensor
    """
    e0, e1, e2, e3 = quaternion[:, 0:1, :], quaternion[:, 1:2, :], quaternion[:, 2:3, :], quaternion[:, 3:, :]
    phi = torch.atan2(2.0 * (e0 * e1 + e2 * e3), (e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0))
    theta = torch.asin(2.0 * (e0 * e2 - e1 * e3))
    psi = torch.atan2(2.0 * (e0 * e3 + e1 * e2), (e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0))

    return phi, theta, psi


def euler_2_quaternion_one(phi, theta, psi):
    """
    Conversion from euler angles to quaternions, in page 259
    Args:
        Euler angles: phi, theta, psi

    Returns:
        quaternions e
    """
    s_phi_2 = np.sin(phi / 2)
    c_phi_2 = np.cos(phi / 2)
    s_theta_2 = np.sin(theta / 2)
    c_theta_2 = np.cos(theta / 2)
    s_psi_2 = np.sin(psi / 2)
    c_psi_2 = np.cos(psi / 2)

    e = np.zeros(4)
    e[0] = c_psi_2 * c_theta_2 * c_phi_2 + s_psi_2 * s_theta_2 * s_phi_2
    e[1] = c_psi_2 * c_theta_2 * s_phi_2 - s_psi_2 * s_theta_2 * c_phi_2
    e[2] = c_psi_2 * s_theta_2 * c_phi_2 + s_psi_2 * c_theta_2 * s_phi_2
    e[3] = s_psi_2 * c_theta_2 * c_phi_2 - c_psi_2 * s_theta_2 * s_phi_2

    return e


def euler_2_quaternion(phi, theta, psi):
    """
    Conversion from euler angles to quaternions, in page 259
    Args:
        Euler angles: phi, theta, psi

    Returns:
        quaternions e
    """
    s_phi_2 = torch.sin(phi / 2)
    c_phi_2 = torch.cos(phi / 2)
    s_theta_2 = torch.sin(theta / 2)
    c_theta_2 = torch.cos(theta / 2)
    s_psi_2 = torch.sin(psi / 2)
    c_psi_2 = torch.cos(psi / 2)

    # e = np.zeros(4)
    e_0 = c_psi_2 * c_theta_2 * c_phi_2 + s_psi_2 * s_theta_2 * s_phi_2
    e_1 = c_psi_2 * c_theta_2 * s_phi_2 - s_psi_2 * s_theta_2 * c_phi_2
    e_2 = c_psi_2 * s_theta_2 * c_phi_2 + s_psi_2 * c_theta_2 * s_phi_2
    e_3 = s_psi_2 * c_theta_2 * c_phi_2 - c_psi_2 * s_theta_2 * s_phi_2

    return e_0, e_1, e_2, e_3


# def euler_2_rotation(phi, theta, psi):
#     """
#     Converts euler angles to rotation matrix (R^i_b, i.e., body to inertial)
#     """
#     # only call sin and cos once for each angle to speed up rendering
#     c_phi = np.cos(phi)
#     s_phi = np.sin(phi)
#     c_theta = np.cos(theta)
#     s_theta = np.sin(theta)
#     c_psi = np.cos(psi)
#     s_psi = np.sin(psi)
#
#     R_roll = np.array([[1, 0, 0], [0, c_phi, s_phi], [0, -s_phi, c_phi]], dtype=object)
#     R_pitch = np.array([[c_theta, 0, -s_theta], [0, 1, 0], [s_theta, 0, c_theta]], dtype=object)
#     R_yaw = np.array([[c_psi, s_psi, 0], [-s_psi, c_psi, 0], [0, 0, 1]], dtype=object)
#     R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 on the book)
#     return R.T  # transpose to return body to inertial


def quaternion_2_rotation(quaternion):
    """
    converts a quaternion attitude to a rotation matrix
    """
    e0, e1, e2, e3 = quaternion[:, 0:1, :], quaternion[:, 1:2, :], quaternion[:, 2:3, :], quaternion[:, 3:, :]

    row_0 = torch.cat(
        (e1**2.0 + e0**2.0 - e2**2.0 - e3**2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)), 2
    )
    row_1 = torch.cat(
        (2.0 * (e1 * e2 + e3 * e0), e2**2.0 + e0**2.0 - e1**2.0 - e3**2.0, 2.0 * (e2 * e3 - e1 * e0)), 2
    )
    row_2 = torch.cat(
        (2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3**2.0 + e0**2.0 - e1**2.0 - e2**2.0), 2
    )
    R = torch.cat((row_0, row_1, row_2), 1)

    # # 为了数值稳定性，暂不启用
    # det = torch.det(R)
    # det = torch.unsqueeze(det, 1)
    # det = torch.unsqueeze(det, 2)
    # R = R / det

    return R


# def rotation_2_quaternion(R):
#     """
#     converts a rotation matrix to a unit quaternion
#     """
#     r11 = R[0][0]
#     r12 = R[0][1]
#     r13 = R[0][2]
#     r21 = R[1][0]
#     r22 = R[1][1]
#     r23 = R[1][2]
#     r31 = R[2][0]
#     r32 = R[2][1]
#     r33 = R[2][2]
#
#     tmp = r11 + r22 + r33
#     if tmp > 0:
#         e0 = 0.5 * np.sqrt(1 + tmp)
#     else:
#         e0 = 0.5 * np.sqrt(((r12 - r21) ** 2 + (r13 - r31) ** 2 + (r23 - r32) ** 2) / (3 - tmp))
#
#     tmp = r11 - r22 - r33
#     if tmp > 0:
#         e1 = 0.5 * np.sqrt(1 + tmp)
#     else:
#         e1 = 0.5 * np.sqrt(((r12 + r21) ** 2 + (r13 + r31) ** 2 + (r23 - r32) ** 2) / (3 - tmp))
#
#     tmp = -r11 + r22 - r33
#     if tmp > 0:
#         e2 = 0.5 * np.sqrt(1 + tmp)
#     else:
#         e2 = 0.5 * np.sqrt(((r12 + r21) ** 2 + (r13 + r31) ** 2 + (r23 + r32) ** 2) / (3 - tmp))
#
#     tmp = -r11 + -22 + r33
#     if tmp > 0:
#         e3 = 0.5 * np.sqrt(1 + tmp)
#     else:
#         e3 = 0.5 * np.sqrt(((r12 - r21) ** 2 + (r13 + r31) ** 2 + (r23 + r32) ** 2) / (3 - tmp))
#
#     return np.array([[e0], [e1], [e2], [e3]])
#
#
# def hat(omega):
#     """
#     vector to skew symmetric matrix associated with cross product
#     """
#     a = omega.item(0)
#     b = omega.item(1)
#     c = omega.item(2)
#
#     omega_hat = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
#     return omega_hat
