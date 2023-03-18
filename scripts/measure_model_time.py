import time
import torch
from quadrotor import MulQuadrotors

if __name__ == "__main__":
    # test model
    num_agent = 1000
    dt_ctl = 0.02
    dt_sim = 0.02

    mul_qd = MulQuadrotors(num_agent, dt_ctl, torch.float64).requires_grad_(False)
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

    count = 0
    time_pre = time.perf_counter()
    count_round_num = 500
    while count < 5000:

        state = sm_mul_qd(state, rate_cmd, dt_sim)
        count += 1

        if count == 500:
            time_now = time.perf_counter()
            print(f"Stable running!")
            # print(f"Time cost for {500} round: {time_now - time_pre} s")
            # print("Time cost for 1 round average: ", (time_now - time_pre) / 500, " s")
            time_pre = time_now

        if count == 500 + 2000:
            time_now = time.perf_counter()
            # print(f"Time cost for {2000} round: {time_now - time_pre} s")
            print("Time cost for 1 round average: ", (time_now - time_pre) / 2000 * 1000, " ms")
            time_pre = time_now

        # if count % count_round_num == 0 and count != 0:
        #     time_now = time.perf_counter()
        #     print(f"Time cost for {count_round_num} round: {time_now - time_pre} s")
        #     print("Time cost for 1 round average: ", (time_now - time_pre) / count_round_num, " s")
        #     time_pre = time_now

    print("Model runs successfully!")

    # # save model
    # sm_mul_qd.save("../models/mul_qd_model.pt")
    # print("Model saves successfully!")
