"""
pid_control
"""
import torch
import torch.nn as nn
from ..tools.saturate import saturate_w_tensor_limit


class PIDControl(nn.Module):
    def __init__(self, num_agent, Ts, dtype=torch.float64, kp=0.0, ki=0.01, kd=0.0, sigma=0.05, limit=0.0):
        super().__init__()
        self.kp = nn.Parameter(kp * torch.ones([num_agent, 1, 1], dtype=dtype), False)
        assert ki != 0, "pid中的ki不能为0！ 若ki为0，请使用pd函数"  # 确保除数不为0
        self.ki = nn.Parameter(ki * torch.ones([num_agent, 1, 1], dtype=dtype), False)
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

        self.integrator = torch.zeros([num_agent, 1, 1])
        self.error_delay_1 = torch.zeros([num_agent, 1, 1])
        self.error_dot_delay_1 = torch.zeros([num_agent, 1, 1])
        self.y_dot = torch.zeros([num_agent, 1, 1])
        self.y_delay_1 = torch.zeros([num_agent, 1, 1])
        self.y_dot_delay_1 = torch.zeros([num_agent, 1, 1])

    def forward(self, y_ref, y):
        if not self.is_init:
            self.integrator = torch.clone(self.zero_param.data)
            self.error_delay_1 = torch.clone(self.zero_param.data)
            self.error_dot_delay_1 = torch.clone(self.zero_param.data)
            self.y_dot = torch.clone(self.zero_param.data)
            self.y_delay_1 = torch.clone(self.zero_param.data)
            self.y_dot_delay_1 = torch.clone(self.zero_param.data)
            self.is_init = True

        # compute the error
        error = y_ref - y
        # update the integrator using trapazoidal rule
        self.integrator = self.integrator + (self.Ts / 2) * (error + self.error_delay_1)
        # update the differentiator
        error_dot = self.a1 * self.error_dot_delay_1 + self.a2 * (error - self.error_delay_1)
        # PID control
        u = self.kp * error + self.ki * self.integrator + self.kd * error_dot
        # saturate PI control at limit
        u_sat = saturate_w_tensor_limit(u, -self.limit, self.limit)
        # integral anti-windup
        # adjust integrator to keep u out of saturation
        flag = torch.abs(self.ki) > 0.0001
        self.integrator = (self.integrator + (self.Ts / self.ki) * (u_sat - u)) * flag + self.integrator * ~flag

        # update the delayed variables
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot

        return u_sat


if __name__ == "__main__":
    pi_func = PIDControl(2000).to("cuda")

    pi_controller = torch.jit.script(pi_func)

    print(1)
