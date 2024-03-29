import torch
from quadrotor import MulQuadrotors

if __name__ == "__main__":
    # test model
    num_agent = 1000
    mul_qd = MulQuadrotors(num_agent, 0.02, torch.float64).requires_grad_(False)
    if torch.cuda.is_available():
        mul_qd = mul_qd.to("cuda")

    sm_mul_qd = torch.jit.script(mul_qd)
    # sm_mul_qd = mul_qd  # debug only

    """
    ego_states: [player, group, HP, north, east, down, phi, theta, psi, ew, ex, ey, ez, vx, vy, vz, u, v, w,
             p, q, r, Va, Vg, alpha, beta, gamma, chi, wn, we, wd]   len=31
    cmd: ["roll_rate_cmd", "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd"]
    """
    state = torch.zeros([num_agent, 31, 1], dtype=torch.float64).to("cuda")
    state[:, 9, 0] = 1.0  # ew

    rate_cmd = torch.zeros([num_agent, 4, 1], dtype=torch.float64).to("cuda")

    state_new = sm_mul_qd(state, rate_cmd, 0.02)

    print("Model runs successfully!")

    # save model
    sm_mul_qd.save("../models/mul_qd_model.pt")
    print("Model saves successfully!")
