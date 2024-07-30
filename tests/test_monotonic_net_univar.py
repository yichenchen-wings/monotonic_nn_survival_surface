from monotonic_nn_surv_surf.core.monotonic_net_univar import LinearMonotonic, MonotonicLayerUnivar, MonotonicNetUnivar
import torch
import numpy as np
import pytest

def test_LinearMonotonic_shape():
    batch_size = 5
    func = LinearMonotonic(in_features=1, out_features=12)
    ts = torch.rand(batch_size, 1)
    out = func(ts)
    
    assert out.shape == (ts.shape[0], 12)


def test_LinearMonotonic_monotone():
    batch_size = 20
    func = LinearMonotonic(in_features=1, out_features=20)
    xs = torch.subtract(
        torch.rand(batch_size, 1),
        0.5
    )
    xs.requires_grad = True
    out = func(xs)
    is_grad_non_neg = []
    for i in out:
        for j in i:
            j.backward(retain_graph=True)
            is_grad_non_neg.append(all(xs.grad >= 0))
    assert all(is_grad_non_neg)
            

def test_MonotonicLayerUnivar_single_shape():
    batch_size = 2
    z = torch.rand(batch_size, 3)
    z0 = torch.rand(batch_size, 5)
    x_mono = torch.rand(batch_size, 1)
    func = MonotonicLayerUnivar(
        input_size=z.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=3, 
        act=torch.relu
    )
    out = func(z=z, z0=z0, x_mono=x_mono)
    assert out.shape == (x_mono.shape[0], 3)


def test_MonotonicLayerUnivar_multi_shape():
    batch_size = 2
    z = torch.zeros(batch_size, 3)
    z0 = torch.rand(batch_size, 5)
    x_mono = torch.rand(batch_size, 1)
    
    z1 = MonotonicLayerUnivar(
        input_size=z.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=3, 
        act=torch.relu
    )(z=z, z0=z0, x_mono=x_mono)
    
    z2 = MonotonicLayerUnivar(
        input_size=z1.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=2, 
        act=torch.relu
    )(z=z1, z0=z0, x_mono=x_mono)

    out = MonotonicLayerUnivar(
        input_size=z2.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=1, 
        act=torch.relu
    )(z=z2, z0=z0, x_mono=x_mono)

    assert out.shape == (x_mono.shape[0], 1)


@pytest.mark.parametrize(
    "seed", 
    [i for i in range(1, 100, 5)]
)
def test_MonotonicLayer_multi_monotone(seed):
    torch.manual_seed(seed)
    batch_size = 30
    z = torch.zeros(batch_size, 3)
    z0 = torch.rand(batch_size, 5)
    z0.requires_grad = True

    x_mono = torch.rand(batch_size, 1)
    x_mono.requires_grad = True

    z1 = MonotonicLayerUnivar(
        input_size=z.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=30, 
        act=torch.relu,
        dropout=seed/150
    )(z=z, z0=z0, x_mono=x_mono)
    
    z2 = MonotonicLayerUnivar(
        input_size=z1.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=20, 
        act=torch.relu,
        dropout=seed/150
    )(z=z1, z0=z0, x_mono=x_mono)

    out = MonotonicLayerUnivar(
        input_size=z2.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=1, 
        act=torch.relu,
        dropout=seed/150
    )(z=z2, z0=z0, x_mono=x_mono)


    is_grad_non_neg = []
    is_grad_non_pos = []
    for i in out:
        i.backward(retain_graph=True)
        is_grad_non_neg.append(all(x_mono.grad >= 0))
    assert all(is_grad_non_neg)
    assert all(is_grad_non_pos)

            
def test_MonotonicNetUnivar_shape():
    batch_size = 2
    x_mono = torch.rand(batch_size, 1)
    net = MonotonicNetUnivar(input_size=3, layer_sizes=[4,5,1])
    out = net(x_mono=x_mono, z=None)

    assert out.shape == (batch_size, 1)


@pytest.mark.parametrize(
    "seed, mid_layer_sizes, dropout", 
    [
        (i, [32,32], i/150) for i in range(1, 100, 10)
    ] + [
        (i, [32, 64, 32], i/300) for i in range(100, 200, 10)
    ] + [
        (i, [32, 64, 32, 32], i/450) for i in range(200, 300, 10)
    ]
)
def test_MonotonicNetUnivar_monotone(seed, mid_layer_sizes, dropout):
    torch.manual_seed(seed)
    batch_size = 100
    x_mono = torch.rand(batch_size, 1, requires_grad=True)
    net = MonotonicNetUnivar(input_size=mid_layer_sizes[0], layer_sizes=mid_layer_sizes[1:] + [1], dropout=dropout)
    out = net(x_mono=x_mono, z=None)

    is_grad_non_neg = []
    for i in out:
        i.backward(retain_graph=True)
        is_grad_non_neg.append(all(x_mono.grad >= 0))

    is_grad_non_neg = np.array(is_grad_non_neg)
                               
    assert all(is_grad_non_neg)


def test_MonotonicNetUnivar_on_gpu():
    if torch.cuda.is_available():
        batch_size = 30
        mid_layer_sizes = [32, 64, 32, 32]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_mono = torch.rand(batch_size, 1).to('cuda')
        net = MonotonicNetUnivar(input_size=mid_layer_sizes[0], layer_sizes=mid_layer_sizes[1:] + [1]).to(device)
        net(x_mono=x_mono, z=None)
    else:
        import warnings
        warnings.warn('Cannot test model on GPU because no GPU (cuda) is detected.')

    with torch.no_grad():
        torch.cuda.empty_cache()

    
