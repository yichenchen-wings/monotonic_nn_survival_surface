from model.monotonic_net import MonotonicLayer, SingleVarPosMonotonic
import torch

def test_SingleVarPosMonotonic_shape():
    batch_size = 5
    func = SingleVarPosMonotonic(output_size=12)
    ts = torch.rand(batch_size, 1)
    out = func(x=ts)
    
    assert out.shape == (ts.shape[0], 12)


def test_SingleVarPosMonotonic_monotone():
    batch_size = 50
    func = SingleVarPosMonotonic(output_size=50)
    xs = torch.rand(batch_size, 1, requires_grad=True)
    out = func(x=xs)
    is_grad_non_neg = []
    for i in out:
        for j in i:
            j.backward(retain_graph=True)
            is_grad_non_neg.append(all(xs.grad >= 0))
    assert all(is_grad_non_neg)
            


