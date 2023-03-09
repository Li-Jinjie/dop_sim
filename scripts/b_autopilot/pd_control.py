"""
pid_control
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import torch
import torch.nn as nn
from ...tools.saturate import saturate_w_tensor_limit


class PDControl(nn.Module):
    def __init__(self, num_agent, Ts, dtype=torch.float64, kp=0.0, kd=0.0, sigma=0.05, limit=0.0):
        super().__init__()
        self.kp = nn.Parameter(kp * torch.ones([num_agent, 1, 1], dtype=dtype), False)
        self.kd = nn.Parameter(kd * torch.ones([num_agent, 1, 1], dtype=dtype), False)
        self.Ts = nn.Parameter(Ts * torch.ones([num_agent, 1, 1], dtype=dtype), False)
        self.limit = nn.Parameter(limit * torch.ones([num_agent, 1, 1], dtype=dtype), False)

        self.a1 = nn.Parameter(
            (2.0 * sigma - Ts) / (2.0 * sigma + Ts) * torch.ones([num_agent, 1, 1], dtype=dtype), False
        )
        self.a2 = nn.Parameter(2.0 / (2.0 * sigma + Ts) * torch.ones([num_agent, 1, 1], dtype=dtype), False)

        # 中间变量。如果把 error_delay_1 写成nn.Parameter，则不能在计算中赋值。如果不设置zero_tensor的量，则第一次运算的时候
        # 无法自动选择在 CPU 还是 GPU 上进行计算。
        self.is_init = False
        self.zero_param = nn.Parameter(torch.zeros([num_agent, 1, 1], dtype=dtype), False)
        self.error_delay_1 = torch.zeros([num_agent, 1, 1])
        self.error_dot_delay_1 = torch.zeros([num_agent, 1, 1])

    def forward(self, y_ref, y):
        if not self.is_init:
            self.error_delay_1 = torch.clone(self.zero_param.data)
            self.error_dot_delay_1 = torch.clone(self.zero_param.data)
            self.is_init = True

        # compute the error
        error = y_ref - y
        # update the differentiator
        error_dot = self.a1 * self.error_dot_delay_1 + self.a2 * (error - self.error_delay_1)
        # PID control
        u = self.kp * error + self.kd * error_dot
        # saturate PID control at limit
        u_sat = saturate_w_tensor_limit(u, -self.limit, self.limit)

        # update the delayed variables
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot

        return u_sat


if __name__ == "__main__":
    pd_func = PDControl(2000).to("cuda")

    pd_controller = torch.jit.script(pd_func)

    print(1)
