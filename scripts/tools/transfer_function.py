"""
transfer function block (SISO)
"""
import torch
import torch.nn as nn


class TransferFunction(nn.Module):
    """
    The discrete implementation of transfer functions. parallel computation version
    Please read the page 16 of addendum
    """

    def __init__(self, num_agent: int, num: torch.Tensor, den: torch.Tensor, Ts: float, dtype=torch.float64):
        super().__init__()
        # expects num and den to be torch.tensor of shape (x, 1, m) and (x, 1, n)
        m = num.shape[2]
        n = den.shape[2]
        # set initial conditions
        self.state = nn.Parameter(torch.zeros([num_agent, n - 1, 1], dtype=dtype), False)
        self.Ts = nn.Parameter(torch.tensor(Ts), False)
        # make the leading coef of den == 1
        num = (den[:, 0:1, 0:1] != 1) * num / den[:, 0:1, 0:1] + (den[:, 0:1, 0:1] == 1) * num
        # # This line will change den, so it should be executed after the change of num.
        den = (den[:, 0:1, 0:1] != 1) * den / den[:, 0:1, 0:1] + (den[:, 0:1, 0:1] == 1) * den

        # if den.item(0) != 1:
        #     num = num / den.item(0)
        #     den = den / den.item(0)  # This line will change den, so it should be executed after the change of num.
        # self.num = num
        # self.den = den
        # set up state space equations in control canonic form
        self.A = nn.Parameter(torch.zeros([num_agent, n - 1, n - 1], dtype=dtype), False)
        self.B = nn.Parameter(torch.zeros([num_agent, n - 1, 1], dtype=dtype), False)
        self.C = nn.Parameter(torch.zeros([num_agent, 1, n - 1], dtype=dtype), False)
        self.B[:, 0:1, 0:1] = 1.0
        if m == n:
            self.D = nn.Parameter(num[:, 0:1, 0:1], False)
            for i in range(0, n - 1):
                self.C[:, 0:1, i : i + 1] = num[:, 0:1, i + 1 : i + 2] - num[:, 0:1, 0:1] * den[:, 0:1, i + 1 : i + 2]
            for i in range(0, n - 1):
                self.A[:, 0:1, i : i + 1] = -den[:, 0:1, i + 1 : i + 2]
            for i in range(1, n - 1):
                self.A[:, i : i + 1, i - 1 : i] = 1.0
        else:
            self.D = nn.Parameter(torch.zeros([num_agent, 1, 1], dtype=dtype), False)
            for i in range(0, m):
                self.C[:, 0:1, n - i - 2 : n - i - 1] = num[:, 0:1, m - i - 1 : m - i]
            for i in range(0, n - 1):
                self.A[:, 0:1, i : i + 1] = -den[:, 0:1, i + 1 : i + 2]
            for i in range(1, n - 1):
                self.A[:, i : i + 1, i - 1 : i] = 1.0

        # print("A=", self.A)
        # print("B=", self.B)
        # print("C=", self.C)
        # print("D=", self.D)

    def forward(self, u: torch.Tensor):
        self.rk4_step(u)
        y = self.h(u)
        return y

    def f(self, state, u):
        xdot = self.A @ state + self.B * u
        return xdot

    def h(self, u):
        y = self.C @ self.state + self.D * u
        return y

    # TODO: 与ode.py合并
    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta 4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)


# if __name__ == "__main__":
#     pd_func = TransferFunction(2000).to("cuda")
#
#     pd_controller = torch.jit.script(pd_func)
#
#     print(1)

# if __name__ == "__main__":
#     import numpy as np
#
#     # instantiate the system
#     Ts = 0.01  # simulation step size
#     # system = (s + 2)/(s^3 + 4s^2 + 5s + 6)
#     # num = np.array([[2, 3, 1]])
#     # den = np.array([[5, 7, 5, 6]])
#
#     dtype = torch.float
#     device = torch.device("cuda:0")
#     num_airplane = 10000
#     id_plane = 999
#
#     # num = np.array([[1, 6, 11, 6]])
#     # den = np.array([[1, 3.5, 5, 3]])
#     num = torch.tensor([[1, 6, 11, 6]], device=device, dtype=dtype)
#     num = num.repeat(num_airplane, 1, 1)
#     den = torch.tensor([[1, 3.5, 5, 3]], device=device, dtype=dtype)
#     den = den.repeat(num_airplane, 1, 1)
#
#     system = TransferFunction(num_airplane, device, dtype, num, den, Ts)
#
#     # main simulation loop
#     sim_time = 0.0
#     time = [sim_time]  # record time for plotting
#     y = system.h(0.0)
#     output = [y[id_plane, 0, 0].item()]  # record output for plotting
#     while sim_time < 10.0:
#         # u = np.random.randn()  # input is white noise
#         u = 1  # step input
#         y = system.update(u)  # update based on current input
#         time.append(sim_time)  # record time for plotting
#         output.append(y[id_plane, 0, 0].item())  # record output for plotting
#         sim_time += Ts  # increment the simulation time
#
#     # plot output vs time
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.set_xticks(np.arange(0, 10, 0.5))
#     plt.plot(time, output)
#     plt.grid()
#     plt.show()
