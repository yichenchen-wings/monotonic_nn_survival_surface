import numpy as np
import torch
from torch import nn

from .monotonic_net_t_plus_tg import MonotonicNetTaddTG
from .monotonic_net_linear_tg import  MonotonicNetTGLinear
from typing import Optional


class SurvSurf2DTaddTG(nn.Module):
    def __init__(
            self,
            z0_size: int,
            hidden_dim: int,
            n_layers: int,
            dropout: float=0,
            zero_at_t0=True
    ):
        super().__init__()
        self.z0_size = z0_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.zero_at_t0 = zero_at_t0
        assert dropout >=0 
        assert dropout <= 1
        self.dropout=dropout

        self.monotone_surface = MonotonicNetTaddTG(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout
        )
    
    def surface_as_prob_0_at_t0(self, ts, gs, xs):
        surf = self.monotone_surface(t=ts, g=gs, z=xs)

        t0 = torch.zeros(*ts.shape, device=ts.device)
        surf_t0 = self.monotone_surface(t=t0, g=gs, z=xs)

        surf_zeroed = surf - surf_t0
        if np.isnan(torch.min(surf_zeroed).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        if self.training:
            surf_zeroed = torch.clamp(surf_zeroed, 0, np.inf) # to cope with dropout

        out = torch.tanh(surf_zeroed)
        return out
    
    def surface_as_prob(self, ts, gs, xs):
        surf = self.monotone_surface(t=ts, g=gs, z=xs)
        out = torch.sigmoid(surf)
        return out
    
    def forward(self, ts, gs, xs=None):
        if self.zero_at_t0:
            return self.surface_as_prob_0_at_t0(ts=ts, gs=gs, xs=xs)
        else:
            return self.surface_as_prob(ts=ts, gs=gs, xs=xs)
    

class SurvSurf2DSigmJoLin(nn.Module): # JointLinear
    def __init__(
            self,
            z0_size: int,
            hidden_dim: int,
            n_layers: int,
            dropout: float=0,
            zero_at_t0=True
    ):
        super().__init__()
        self.z0_size = z0_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.zero_at_t0 = zero_at_t0
        assert dropout >= 0 
        assert dropout < 1
        self.dropout=dropout

        if self.zero_at_t0:
            sigmoid_act = False
        else:
            sigmoid_act = True
        
        self.monotonice_surfaces_2d = MonotonicNetTGLinear(
            input_size=z0_size,
            layer_sizes=[self.hidden_dim]*self.n_layers + [1],
            dropout=dropout,
            sigmoid_act=sigmoid_act
        )
    
    def surface_as_prob_0_at_t0(self, ts, gs, xs):
        surf = self.monotonice_surfaces_2d(t=ts, g=gs, z=xs)

        t0 = torch.zeros(*ts.shape, device=ts.device)
        surf_t0 = self.monotonice_surfaces_2d(t=t0, g=gs, z=xs)

        surf_zeroed = surf - surf_t0
        if np.isnan(torch.min(surf_zeroed).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        if self.training:
            surf_zeroed = torch.clamp(surf_zeroed, 0, np.inf) # to cope with dropout

        out = torch.tanh(surf_zeroed)
        return out
    
    def surface_as_prob(self, ts, gs, xs):
        surf = self.monotonice_surfaces_2d(t=ts, g=gs, z=xs)
        out = torch.sigmoid(surf)
        return out
    
    def forward(self, ts, gs, xs=None):
        if self.zero_at_t0:
            return self.surface_as_prob_0_at_t0(ts=ts, gs=gs, xs=xs)
        else:
            return self.surface_as_prob(ts=ts, gs=gs, xs=xs)
 
        