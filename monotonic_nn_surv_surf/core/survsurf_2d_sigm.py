import numpy as np
import torch
from torch import nn
from .monotonic_net_univar import MonotonicNetUnivar, LinearMonotonic


class SurvSurf2DSigm(nn.Module):
    def __init__(
            self,
            z0_size: int,
            hidden_dim: int,
            n_layers: int,
    ):
        super().__init__()
        self.z0_size = z0_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.t_intera_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers
        )
        self.g_intera_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers
        )

        self.t_main_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers
        )
        self.g_main_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers
        )
        self.out_layer = LinearMonotonic(in_features=self.hidden_dim, out_features=1)
        self.act_fn = torch.tanh

    def monoton_surface(self, ts, gs, xs):
        t_for_intera = self.t_intera_block(x_mono=ts, z=xs)
        g_for_intera = self.g_intera_block(x_mono=-gs, z=xs)

        t_main = self.t_main_block(x_mono=ts, z=xs)
        g_main = self.g_main_block(x_mono=-gs, z=xs)

        surfs = torch.sigmoid(t_for_intera)*torch.sigmoid(g_for_intera) + torch.sigmoid(t_main) + torch.sigmoid(g_main)
        out = self.out_layer(surfs)/self.hidden_dim
        return out
    
    def forward(self, ts, gs, xs=None):
        assert torch.all(gs > 0)
        t0 = torch.zeros(*ts.shape, device=ts.device)

        surf = self.monoton_surface(ts, gs, xs)
        surf_t0 = self.monoton_surface(t0, gs, xs)

        surf_zeroed = surf - surf_t0
        if np.isnan(torch.min(surf_zeroed).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        assert torch.all(-1e-2 < surf_zeroed), f"{torch.min(surf_zeroed)}"
        surf_zeroed = torch.clamp(surf_zeroed, 0, np.inf)
        out = self.act_fn(surf_zeroed)

        return out
        