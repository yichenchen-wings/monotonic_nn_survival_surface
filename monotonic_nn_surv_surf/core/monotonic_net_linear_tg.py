import torch
import torch.nn as nn

class LinearMonotonic(nn.Linear):
    def forward(self, input):
        weight = torch.abs(self.weight)
        assert torch.all(weight >= 0), f'{weight}'
        return nn.functional.linear(input, weight, self.bias)

class MonotonicLayerUnivar(nn.Module):
    def __init__(self, input_size, z0_input_size, output_size, act, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.z0_input_size = z0_input_size
        self.output_size = output_size

        self.single_var_monotone_pos = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
            bias=False
        )

        self.act = act #activation function

        self.A = LinearMonotonic(self.input_size, output_size, bias=False) #non-neg
        self.B = nn.Linear(self.z0_input_size, output_size, bias=True) #whatever sign
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, z0, x_mono):
        z = self.dropout(z)
        assert z.shape == (z.shape[0], self.input_size), f"z.shape={z.shape},self.input_size={self.input_size}"
        assert x_mono.shape == (z.shape[0], 1)

        alpha_x_mono = self.single_var_monotone_pos(x_mono)

        Az = self.A(z)
        Bz0 = self.B(z0)
        
        z_new = self.act(
            alpha_x_mono 
            + Az 
            + Bz0
        )
        assert z_new.shape == (z.shape[0], self.output_size)
        return z_new
    
class MonotonicLayerTGLinear(nn.Module):
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
        self.g_monotone_pos = LinearMonotonic(
            in_features=1,
            out_features=output_size, 
            bias=False
        )

        self.act = act #activation function

        self.A = LinearMonotonic(self.input_size, output_size, bias=False) #non-neg
        self.B = nn.Linear(self.z0_input_size, output_size, bias=True) #whatever sign
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, z0, t, g):
        z = self.dropout(z)
        assert z.shape == (z.shape[0], self.input_size), f"z.shape={z.shape},self.input_size={self.input_size}"
        assert t.shape == (z.shape[0], 1)
        assert g.shape == (g.shape[0], 1)

        alpha_t = self.t_monotone_pos(t)
        gamma_g = self.g_monotone_pos(-g)

        Az = self.A(z)
        Bz0 = self.B(z0)
        
        z_new = self.act(
            alpha_t/self.output_size
            + gamma_g/self.output_size
            + Az/self.output_size
            + Bz0
        )
        assert z_new.shape == (z.shape[0], self.output_size)
        return z_new
    

class MonotonicNetUnivar(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        n_layers = len(self.layer_sizes)

        self.layers = nn.ModuleList([])
        act = nn.Identity() if n_layers == 1 else nn.Tanh()
        layer = MonotonicLayerUnivar(
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

            layer = MonotonicLayerUnivar(
                input_size=self.layer_sizes[i],
                z0_input_size=self.input_size,
                output_size=self.layer_sizes[i+1],
                act=act,
                dropout=dropout
            )

            self.layers.append(layer)

    def forward(self, x_mono, z=None):
        assert x_mono.shape == (x_mono.shape[0], 1)

        if z is None:
            z = torch.zeros(x_mono.shape[0], self.input_size, device=x_mono.device)

        z0 = z.clone()

        for layer in self.layers:
            z = layer(z=z, z0=z0, x_mono=x_mono)

        assert z.shape == (x_mono.shape[0], self.layer_sizes[-1])
        return z
    

class MonotonicNetTGLinear(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        n_layers = len(self.layer_sizes)

        self.layers = nn.ModuleList([])
        act = nn.Identity() if n_layers == 1 else nn.ELU()
        layer = MonotonicLayerTGLinear(
            input_size=self.input_size,
            z0_input_size=self.input_size,
            output_size=self.layer_sizes[0],
            act=act,
            dropout=dropout
        )

        self.layers.append(layer)
        for i in range(n_layers - 1):
            is_last = (i == n_layers - 2)
            act = nn.Identity() if is_last else nn.ELU()

            layer = MonotonicLayerTGLinear(
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

        for layer in self.layers:
            z = layer(z=z, z0=z0, t=t, g=g)

        assert z.shape == (t.shape[0], self.layer_sizes[-1])
        return z