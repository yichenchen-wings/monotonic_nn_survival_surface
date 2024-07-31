import numpy as np
import torch
from torch import nn
from .monotonic_net_linear_tg import MonotonicNetUnivar, MonotonicNetTGLinear
from typing import Optional


class SurvSurf2DSigm(nn.Module):
    def __init__(
            self,
            z0_size: int,
            hidden_dim: int,
            n_layers: int,
            dropout: float=0
    ):
        super().__init__()
        self.z0_size = z0_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        assert dropout >=0 
        assert dropout <= 1
        self.dropout=dropout

        self.t_intera_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout
        )
        self.g_intera_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout
        )

        self.t_main_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout
        )
        self.g_main_block = MonotonicNetUnivar(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout
        )
        self.act_fn = torch.nn.Tanh()

    def monoton_surface(self, ts, gs, xs):
        t_for_intera = self.t_intera_block(x_mono=ts, z=xs)
        g_for_intera = self.g_intera_block(x_mono=-gs, z=xs)

        t_main = self.t_main_block(x_mono=ts, z=xs)
        g_main = self.g_main_block(x_mono=-gs, z=xs)

        out = torch.sigmoid(t_for_intera)*torch.sigmoid(g_for_intera) + torch.sigmoid(t_main) + torch.sigmoid(g_main)
        return out
    
    def forward(self, ts, gs, xs=None):
        assert torch.all(gs > 0)
        t0 = torch.zeros(*ts.shape, device=ts.device)

        surf = self.monoton_surface(ts, gs, xs)
        surf_t0 = self.monoton_surface(t0, gs, xs)

        surf_zeroed = surf - surf_t0
        if np.isnan(torch.min(surf_zeroed).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        if self.training:
            surf_zeroed = torch.clamp(surf_zeroed, 0, np.inf) # to cope with dropout
        out = self.act_fn(surf_zeroed)

        return out
    

class SurvSurf2DSigmJoLin(nn.Module): # JointLinear
    def __init__(
            self,
            z0_size: int,
            hidden_dim: int,
            n_layers: int,
            dropout: float=0
    ):
        super().__init__()
        self.z0_size = z0_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        assert dropout >=0 
        assert dropout <= 1
        self.dropout=dropout

        self.monotonice_surfaces_2d = MonotonicNetTGLinear(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout
        )
    
        self.act_fn = torch.tanh
    def forward(self, ts, gs, xs=None):
        assert torch.all(gs > 0)
        t0 = torch.zeros(*ts.shape, device=ts.device)

        surf = self.monotonice_surfaces_2d(t=ts, g=gs, z=xs)
        surf_t0 = self.monotonice_surfaces_2d(t=t0, g=gs, z=xs)

        surf_zeroed = surf - surf_t0
        if np.isnan(torch.min(surf_zeroed).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        if self.training:
            surf_zeroed = torch.clamp(surf_zeroed, 0, np.inf) # to cope with dropout
        out = self.act_fn(surf_zeroed)

        return out
        