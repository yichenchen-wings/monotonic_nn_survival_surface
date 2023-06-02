from model.monotonic_net import MonotonicLayer, SingleVarPosMonotonic
import torch

def test_SingleVarPosMonotonic_shape():
    batch_size = 5
    func = SingleVarPosMonotonic(output_size=12)
    ts = torch.rand(batch_size, 1)
    out = func(x=ts)
    
    assert out.shape == (ts.shape[0], 12)


def test_SingleVarPosMonotonic_monotone():
    batch_size = 20
    func = SingleVarPosMonotonic(output_size=20)
    xs = torch.subtract(
        torch.rand(batch_size, 1),
        0.5
    )
    xs.requires_grad = True
    out = func(x=xs)
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


def test_MonotonicLayer_multi_monotone():
    batch_size = 2
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
    for i in out:
        i.backward(retain_graph=True)
        is_grad_non_neg.append(all(ts.grad >= 0))
        is_grad_non_neg.append(all(gs.grad <= 0))
    assert all(is_grad_non_neg)

            
