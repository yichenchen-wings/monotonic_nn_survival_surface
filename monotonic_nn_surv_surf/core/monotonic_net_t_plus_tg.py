import torch
import torch.nn as nn


class LinearMonotonic(nn.Linear):
    def forward(self, input):
        weight = torch.abs(self.weight)
        assert torch.all(weight >= 0), f'{weight}'
        return nn.functional.linear(input, weight, self.bias)
    
class ElemwiseMult1DPos(nn.Module):
    def __init__(self, output_size=64, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.empty(self.output_size, **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, x):
        weight = torch.abs(self.weight)
        y = weight*x
        assert torch.all(weight >= 0), f'{weight}'
        assert x.shape == (x.shape[0], self.output_size), f"x.shape={x.shape}"
        assert y.shape == (y.shape[0], self.output_size), f"y.shape={y.shape}"
        return y

class MonotonicLayerTaddTG(nn.Module):
    def __init__(self, input_size, z0_input_size, output_size, act, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.z0_input_size = z0_input_size
        self.output_size = output_size

        self.t_monotone_pos = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
            bias=False
        )
        
        self.t_monotone_pos_in_tg = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
            bias=True
        )

        self.g_monotone_pos_in_tg = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
            bias=True
        )

        self.act = act #activation function

        self.A = LinearMonotonic(self.input_size, output_size, bias=True) #non-neg
        self.G = ElemwiseMult1DPos(output_size) #non-neg
        self.B = nn.Linear(self.z0_input_size, output_size, bias=True) #whatever sign
        self.C = nn.Linear(self.input_size, output_size, bias=False) #whatever sign
        self.act_z0 = torch.nn.Hardsigmoid()
        
        self.dropout = nn.Dropout(dropout)

    def sigm_fun_in_t(self, t):
            return torch.sigmoid(self.t_monotone_pos_in_tg(t))

    def sigm_fun_in_g(self, g):
        g_mono_down = self.g_monotone_pos_in_tg(-g)
        g_mono_down_pos = torch.sigmoid(g_mono_down)
        return g_mono_down_pos
    
    def forward(self, z, z0, func_z0, t, g):
        z = self.dropout(z)
        assert z.shape == (z.shape[0], self.input_size), f"z.shape={z.shape},self.input_size={self.input_size}"
        assert t.shape == (z.shape[0], 1)
        assert g.shape == (g.shape[0], 1)
        assert func_z0.shape == (z.shape[0], self.input_size)


        alpha_t = self.t_monotone_pos(t)

        g_mono_down_pos = self.sigm_fun_in_g(g)

        t0 = torch.zeros(*t.shape, device=t.device)
        t_mono_up_zero_at_t0 =  self.sigm_fun_in_t(t) - self.sigm_fun_in_t(t0)
        gamma_tg = self.G(t_mono_up_zero_at_t0*g_mono_down_pos)

        Az = self.A(z)
        Bz0 = self.B(z0)
        C_func_z0 = self.C(func_z0)
        func_z0 = self.act_z0(C_func_z0 + Bz0) - 0.5
        
        z_new = self.act(
            alpha_t
            + gamma_tg
            + Az
            + func_z0
        )
        assert z_new.shape == (z.shape[0], self.output_size)
        return z_new, func_z0

    

class MonotonicNetTaddTG(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout=0, sigmoid_act=None):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        n_layers = len(self.layer_sizes)

        self.layers = nn.ModuleList([])
        
        act = nn.Identity() if n_layers == 1 else nn.Tanh()
        layer = MonotonicLayerTaddTG(
            input_size=self.input_size,
            z0_input_size=self.input_size,
            output_size=self.layer_sizes[0],
            act=act,
            dropout=dropout
        )

        self.layers.append(layer)
        for i in range(n_layers - 1):
            is_last = (i == n_layers - 2)
            
            act = nn.Identity() if is_last else nn.Tanh()

            layer = MonotonicLayerTaddTG(
                input_size=self.layer_sizes[i],
                z0_input_size=self.input_size,
                output_size=self.layer_sizes[i+1],
                act=act,
                dropout=dropout
            )

            self.layers.append(layer)

    def forward(self, t, g, z=None):
        assert t.shape == (t.shape[0], 1)
        assert g.shape == (g.shape[0], 1)

        if z is None:
            z = torch.zeros(t.shape[0], self.input_size, device=t.device)

        z0 = z.clone()
        func_z0 = torch.zeros(t.shape[0], self.input_size, device=t.device)
        for layer in self.layers:
            z, func_z0 = layer(z=z, z0=z0, func_z0=func_z0, t=t, g=g)

        assert z.shape == (t.shape[0], self.layer_sizes[-1])
        return z