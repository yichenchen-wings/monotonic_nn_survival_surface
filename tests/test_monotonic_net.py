from monotonic_nn_surv_surf.core.monotonic_net import MonotonicLayer, LinearMonotonic, MonotonicNet
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
            

def test_MonotonicLayer_single_shape():
    batch_size = 2
    z = torch.rand(batch_size, 3)
    z0 = torch.rand(batch_size, 5)
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    func = MonotonicLayer(
        input_size=z.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=3, 
        act=torch.relu
    )
    out = func(z=z, z0=z0, t=ts, g=gs)
    assert out.shape == (ts.shape[0], 3)


def test_MonotonicLayer_multi_shape():
    batch_size = 2
    z = torch.zeros(batch_size, 3)
    z0 = torch.rand(batch_size, 5)
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    
    z1 = MonotonicLayer(
        input_size=z.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=3, 
        act=torch.relu
    )(z=z, z0=z0, t=ts, g=gs)
    
    z2 = MonotonicLayer(
        input_size=z1.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=2, 
        act=torch.relu
    )(z=z1, z0=z0, t=ts, g=gs)

    out = MonotonicLayer(
        input_size=z2.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=1, 
        act=torch.relu
    )(z=z2, z0=z0, t=ts, g=gs)

    assert out.shape == (ts.shape[0], 1)


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

    ts = torch.rand(batch_size, 1)
    ts.requires_grad = True

    gs = torch.rand(batch_size, 1)
    gs.requires_grad = True

    z1 = MonotonicLayer(
        input_size=z.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=30, 
        act=torch.relu
    )(z=z, z0=z0, t=ts, g=gs)
    
    z2 = MonotonicLayer(
        input_size=z1.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=20, 
        act=torch.relu
    )(z=z1, z0=z0, t=ts, g=gs)

    out = MonotonicLayer(
        input_size=z2.shape[-1], 
        z0_input_size=z0.shape[-1], 
        output_size=1, 
        act=torch.relu
    )(z=z2, z0=z0, t=ts, g=gs)


    is_grad_non_neg = []
    is_grad_non_pos = []
    for i in out:
        i.backward(retain_graph=True)
        is_grad_non_neg.append(all(ts.grad >= 0))
        is_grad_non_pos.append(all(gs.grad <= 0))
    assert all(is_grad_non_neg)
    assert all(is_grad_non_pos)

            
def test_MonotonicNet_shape():
    batch_size = 2
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    net = MonotonicNet(latent_sizes=[3,4,5,1])
    out = net(t=ts, g=gs, z=None)

    assert out.shape == (batch_size, 1)


@pytest.mark.parametrize(
    "seed, mid_layer_sizes", 
    [
        (i, [32,32]) for i in range(1, 100, 10)
    ] + [
        (i, [32, 64, 32]) for i in range(100, 200, 10)
    ] + [
        (i, [32, 64, 32, 32]) for i in range(200, 300, 10)
    ]
)
def test_MonotonicNet_monotone(seed, mid_layer_sizes):
    torch.manual_seed(seed)
    batch_size = 100
    ts = torch.rand(batch_size, 1, requires_grad=True)
    gs = torch.rand(batch_size, 1, requires_grad=True)
    net = MonotonicNet(latent_sizes=mid_layer_sizes + [1])
    out = net(t=ts, g=gs, z=None)

    is_grad_non_neg = []
    is_grad_non_pos = []
    for i in out:
        i.backward(retain_graph=True)
        is_grad_non_neg.append(all(ts.grad >= 0))
        is_grad_non_pos.append(all(gs.grad <= 0))

    is_grad_non_neg = np.array(is_grad_non_neg)
    is_grad_non_pos = np.array(is_grad_non_pos)
                               
    assert all(is_grad_non_neg)
    assert all(is_grad_non_pos) 

@pytest.mark.parametrize(
    "seed", 
    [i for i in range(1, 100, 5)]
)
def test_MonotonicNet_zero(seed):
    torch.manual_seed(seed)
    batch_size = 30
    ts = torch.zeros(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    net = MonotonicNet(latent_sizes=[10, 10, 1])
    out = net(t=ts, g=gs, z=None)

    is_zero = []
    for i in out:
        is_zero.append(i == 0)
    assert all(is_zero)



@pytest.mark.parametrize(
    "seed, mid_layer_sizes", 
    [
        (i, [32,32]) for i in range(1, 100, 10)
    ] + [
        (i, [32, 64, 32]) for i in range(100, 200, 10)
    ] + [
        (i, [32, 64, 32, 32]) for i in range(200, 300, 10)
    ]
)
def test_MonotonicNet_non_neg(seed, mid_layer_sizes):
    torch.manual_seed(seed)
    batch_size = 30
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    net = MonotonicNet(latent_sizes=mid_layer_sizes + [1])
    out = net(t=ts, g=gs, z=None)

    is_pos = []
    for i in out:
        is_pos.append(i > 0)
    assert any(is_pos)


def test_MonotonicNet_on_gpu():
    if torch.cuda.is_available():
        batch_size = 30
        mid_layer_sizes = [32, 64, 32, 32]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ts = torch.rand(batch_size, 1).to('cuda')
        gs = torch.rand(batch_size, 1).to('cuda')
        net = MonotonicNet(latent_sizes=mid_layer_sizes + [1]).to(device)
        net(t=ts, g=gs, z=None)
    else:
        import warnings
        warnings.warn('Cannot test model on GPU because no GPU (cuda) is detected.')

    with torch.no_grad():
        torch.cuda.empty_cache()

    
