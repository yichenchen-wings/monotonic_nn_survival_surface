import torch
import numpy as np
import torch.nn as nn


class SingleVarPosMonotonic(nn.Module):

    def __init__(self, output_size=64):
        super().__init__()
        self.output_size = output_size

        self.b =  nn.Parameter(torch.rand(1))
        self.w_ = nn.Parameter(torch.rand(1, output_size))

    @property
    def w(self):
        self.w_.data.clamp_(min=0)
        return self.w_


    def transform(self, x):
        assert x.shape == (len(x), 1), f"{x.shape=}"
        return torch.matmul(x+self.b, self.w)

    def forward(self, x):
        y = self.transform(x)
        assert x.shape == (x.shape[0], 1), f"{x.shape=}"
        assert y.shape == (y.shape[0], self.output_size), f"{y.shape=}"
        return torch.exp(y)


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
        alpha_t_vs_gamma_g = alpha_t/gamma_g
        theta_t_vs_g = self.single_var_monotone_pos(t/g)
        Az = self.A(z)
        Bz0 = self.B(z0)
        z_new = self.act(alpha_t_vs_gamma_g + theta_t_vs_g + Az + Bz0)


        assert z_new.shape == (z.shape[0], self.output_size)
        return z_new
    

class MonotonicNet(nn.Module):

    def __init__(self, latent_sizes):
        super().__init__()
        self.sizes = latent_sizes

        self.layers = nn.ModuleList([])

        for i in range(len(self.sizes) - 1):
            is_last = (i == len(self.sizes) - 2)
            act = nn.Identity() if is_last else nn.Tanh()

            layer = MonotonicLayer(
                input_size=self.sizes[i],
                z0_input_size=self.sizes[0],
                output_size=self.sizes[i+1],
                act=act,
            )

            self.layers.append(layer)


    def _adjust_towards_zero(self, z, z0, t, g):
        z = z - self(t=torch.zeros(*t.shape), g=g, z=z0, survival=False)
        if np.isnan(torch.min(z).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        assert torch.all(-1e-2 < z), f"{torch.min(z)=}"

        z = torch.clamp(z, 0, np.inf)
        return z


    def forward(self, t, g, z=None, survival=True):
        assert t.shape == (t.shape[0], 1)
        assert torch.all(t >= 0)

        if z is None:
            z = torch.zeros(t.shape[0], self.sizes[0], device=t.device)

        z0 = z.clone()

        for layer in self.layers:
            z = layer(z=z, z0=z0, t=t, g=g)

        if survival:
            z = self._adjust_towards_zero(z=z, z0=z0, t=t, g=g)

        assert z.shape == (t.shape[0], self.sizes[-1])
        return z