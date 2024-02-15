import torch
import numpy as np
import torch.nn as nn
import math
    
class LinearMonotonic(nn.Linear):
    def forward(self, input):
        weight_sqr = torch.square(self.weight)
        assert torch.all(weight_sqr >= 0), f'{weight_sqr}'
        return nn.functional.linear(input, weight_sqr, self.bias)

class ElemwiseMult1D(nn.Module):
    def __init__(self, output_size=64, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.empty(self.output_size, **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, x):
        weight_sqr = torch.square(self.weight)
        y = weight_sqr*x
        assert torch.all(weight_sqr >= 0), f'{weight_sqr}'
        assert x.shape == (x.shape[0], self.output_size), f"x.shape={x.shape}"
        assert y.shape == (y.shape[0], self.output_size), f"y.shape={y.shape}"
        return y

class MonotonicLayer(nn.Module):

    def __init__(self, input_size, z0_input_size, output_size, act):
        super().__init__()
        self.input_size = input_size
        self.z0_input_size = z0_input_size
        self.output_size = output_size

        self.single_var_monotone_pos_t = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
        )
        self.single_var_monotone_pos_g = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
        )

        self.act = act #activation function
        #self.pos_fn = torch.nn.Softplus()

        self.A = self._get_A(input_size, output_size) #non-neg
        #self.A_intera_g = self._get_A_intera(input_size, output_size) #non-neg
        #self.A_intera_t = self._get_A_intera(input_size, output_size) #non-neg
        self.B = self._get_B(z0_input_size, output_size) #whatever sign
        self.G = self._get_G(output_size) #non-neg
        

    def _get_A(self, input_size, output_size):
        A = LinearMonotonic(input_size, output_size, bias=False) # just scale
        return A
    
    def _get_A_intera(self, input_size, output_size):
        A_intera = LinearMonotonic(input_size, output_size, bias=False) #just scale
        return A_intera

    def _get_B(self, input_size, output_size):
        B = nn.Linear(input_size, output_size, bias=True) #scale and add bias
        return B

    def _get_G(self, output_size):
        G = ElemwiseMult1D(output_size)
        return G

    def forward(self, z, z0, t, g):
        assert z.shape == (z.shape[0], self.input_size), f"z.shape={z.shape},self.input_size={self.input_size}"
        assert t.shape == (z.shape[0], 1)
        assert g.shape == (g.shape[0], 1)
        assert torch.all(t >= 0)
        assert torch.all(g > 0)

        alpha_t = self.single_var_monotone_pos_t(t)

        t_for_g = self.single_var_monotone_pos_g(t)
        t_0 = torch.zeros(t.shape[0], 1, device=t.device)
        t_for_g_t0 = self.single_var_monotone_pos_g(t_0)
        s_func = torch.nn.Softsign()
        sigm_t = s_func(t_for_g) - s_func(t_for_g_t0) # to ensure sigm_t is 0 at t=0
        t_vs_g = sigm_t/g
        G_gamma_t_vs_g = self.G(t_vs_g)

        Az = self.A(z)
       # Az_for_t = self.pos_fn(self.A_intera_t(z))
       # Az_for_g = self.pos_fn(self.A_intera_g(z))
        Bz0 = self.B(z0)
        
        z_new = self.act(
            #Az_for_t*t 
            alpha_t 
            #+ Az_for_g*t_vs_g 
            + G_gamma_t_vs_g 
            + Az 
            + Bz0
        )


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
        t0 = torch.zeros(*t.shape, device=t.device)
        z_t0 = self(t=t0, g=g, z=z0, survival=False)
        z_adj = z - z_t0
        if np.isnan(torch.min(z_adj).item()):
            raise ValueError("Found a nan in one of MonotonicNet's activations.")
        assert torch.all(-1e-2 < z_adj), f"{torch.min(z_adj)}"
        z_adj = torch.clamp(z_adj, 0, np.inf)
        
        return z_adj


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