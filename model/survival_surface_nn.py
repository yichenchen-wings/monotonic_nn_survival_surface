import torch.nn as nn
import torch 

class SurvivalSurface(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.M = net
    
    def forward(self, ts, gs, xs=None):
        net_output = self.M(t=ts, g=gs, z=xs)

        S_t = torch.subtract(1, torch.exp(-net_output))
        return S_t
    