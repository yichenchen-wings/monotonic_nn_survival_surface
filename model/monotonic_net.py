import torch
import numpy as np
import torch.nn as nn


class SingleVarPosMonotonic(nn.Module):

    def __init__(self, output_size=64):
        super().__init__()
        self.output_size = output_size
        #self.w_ = nn.Parameter(torch.rand(1, output_size)) old code
        #self.b  = nn.Parameter(torch.rand(1, output_size)) old code

        self.w_ = nn.Parameter(torch.rand(1, output_size))# check size

    @property
    def w(self):
        self.w_.data.clamp_(min=0)
        return self.w_


    def transform(self, x):
        assert x.shape == (len(x), 1), f"{x.shape=}"
        return torch.matmul(x, self.w) 


    def forward(self, x):
        y = self.transform(x)
        assert x.shape == (x.shape[0], 1), f"{x.shape=}"
        assert y.shape == (y.shape[0], self.output_size), f"{y.shape=}"
        return y


class MonotonicLayer(nn.Module):

    def __init__(self, input_size, z0_input_size, output_size, act):
        super().__init__()
        self.input_size = input_size
        self.z0_input_size = z0_input_size
        self.output_size = output_size

        self.single_var_monotone_pos = SingleVarPosMonotonic(
            output_size=output_size, 
        )

        self.act = act #activation function

        self.A = self._get_A(input_size, output_size)
        self.B = self._get_B(z0_input_size, output_size)


    def _get_A(self, input_size, output_size):
        A = nn.Linear(input_size, output_size, bias=True)
        A.weight.data = A.weight.data.abs()
        return A

    def _get_B(self, input_size, output_size):
        B = nn.Linear(input_size, output_size)
        return B

    @torch.no_grad()
    def _clamp_weights(self):
        self.A.weight.data.clamp_(0)


    def forward(self, z, z0, t, g):
        assert z.shape == (z.shape[0], self.input_size), f"{z.shape=}, {z.shape=}, {(z.shape[0], self.input_size)=}"
        assert t.shape == (z.shape[0], 1)
        assert g.shape == (g.shape[0], 1)
        assert torch.all(t >= 0)
        assert torch.all(g > 0)

        self._clamp_weights()
        alpha_t = self.single_var_monotone_pos(t)
        gamma_g = self.single_var_monotone_pos(g)
        Az = self.A(z)
        Bz0 = self.B(z0)
        z_new = self.act(alpha_t - gamma_g + Az + Bz0)


        assert z_new.shape == (z.shape[0], self.output_size)
        return z_new