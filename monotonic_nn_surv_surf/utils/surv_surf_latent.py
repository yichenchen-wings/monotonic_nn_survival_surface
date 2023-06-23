import torch.nn as nn
from ..core.monotonic_net import MonotonicNet
from ..core.survival_surface_nn import SurvivalSurface

class LatentFeatFC(nn.Module):
    def __init__(self, input_size, output_size, neurons_per_layer=[], dropout_p=0.2):
        super().__init__()
        self.sizes = [input_size] + neurons_per_layer + [output_size]
        n_layers = len(self.sizes)
        self.dropout_p = dropout_p

        self.layers = nn.ModuleList([])

        for i, size in enumerate(self.sizes):
            i_next = i+1
            if i_next == n_layers:
                break
            
            norm = nn.BatchNorm1d(num_features=size)
            self.layers.append(norm)    
            
            drop = nn.Dropout(p=self.dropout_p, inplace=False)
            self.layers.append(drop)

            layer = nn.Linear(size, self.sizes[i_next])
            self.layers.append(layer)

            act = nn.ReLU()
            self.layers.append(act)
            


    def forward(self, xs):
        assert xs.shape[-1] == self.sizes[0]
        out = xs.clone()
        for layer in self.layers:
            out = layer(out)
        assert out.shape[-1] == self.sizes[-1]
        return out


class SurvSurfLatent(nn.Module):
    def __init__(self, mono_net_sizes, latent_feat_transformer):
        super().__init__()
        
        assert mono_net_sizes[-1] == 1
        assert mono_net_sizes[0] == latent_feat_transformer.sizes[-1]
        self.mono_net_sizes = mono_net_sizes
        self.latent_feat_transformer = latent_feat_transformer
        mono_net = MonotonicNet(latent_sizes=mono_net_sizes)
        self.surv_surf = SurvivalSurface(net=mono_net)

    def forward(self, ts, gs, xs):
        z0 = self.latent_feat_transformer(xs)
        S_tg = self.surv_surf(ts=ts, gs=gs, xs=z0)
        return S_tg