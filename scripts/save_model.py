import torch
from quadrotor import MulQuadrotors

if __name__ == "__main__":
    # test model
    num_agent = 10
    mul_qd = MulQuadrotors(num_agent, 0.02, torch.float64).requires_grad_(False)
    if torch.cuda.is_available():
        mul_qd = mul_qd.to("cuda")

    sm_mul_qd = torch.jit.script(mul_qd)

    """
    ego_states: [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
             p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we]   len=30
    cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
    """
    state = torch.zeros([num_agent, 30, 1], dtype=torch.float64)
    state[:, 9, 0] = 1.0  # ew

    rate_cmd = torch.zeros([num_agent, 4, 1], dtype=torch.float64)

    state_new = sm_mul_qd

    # save model
    sm_mul_qd.save("../models/mul_qd_model.pt")

