from model.monotonic_net import MonotonicNet
from model.survival_surface_nn import SurvivalSurface
import torch
import numpy as np

def test_SurvivalSurface_shape():
    batch_size = 10
    feat_size = 2
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    xs = torch.rand(batch_size, feat_size)
    net = MonotonicNet([feat_size,4,6,1])

    surf = SurvivalSurface(net=net)
    out = surf(ts=ts, gs=gs, xs=xs)
    assert out.shape == (batch_size, 1)


def test_SurvivalSurface_monotone():
    batch_size = 100
    feat_size = 2
    ts = torch.rand(batch_size, 1) * 100
    gs = torch.rand(batch_size, 1) * 100
    ts.requires_grad = True
    gs.requires_grad = True

    xs = torch.rand(batch_size, feat_size)*100 - 50
    net = MonotonicNet([feat_size,64,64,1])

    surf = SurvivalSurface(net=net)
    out = surf(ts=ts, gs=gs, xs=xs)

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


def test_SurvivalSurface_range():
    batch_size = 30
    feat_size = 2
    ts = torch.zeros(batch_size, 1) * 100
    gs = torch.rand(batch_size, 1) * 100
    xs = torch.rand(batch_size, feat_size) * 100 - 50
    net = MonotonicNet([feat_size,10,10,1])

    surf = SurvivalSurface(net=net)
    out = surf(ts=ts, gs=gs, xs=xs)

    is_zero = []
    for i in out:
        is_zero.append(all(i == 0))
    is_zero = np.array(is_zero)
    assert all(is_zero)


def test_SurvivalSurface_max():
    batch_size = 30
    feat_size = 2
    ts = torch.rand(batch_size, 1) * 1000
    gs = torch.rand(batch_size, 1) * 1000
    xs = torch.rand(batch_size, feat_size)*1000 - 500
    net = MonotonicNet([feat_size,10,10,1])

    surf = SurvivalSurface(net=net)
    out = surf(ts=ts, gs=gs, xs=xs)

    is_at_most_one = []
    for i in out:
        is_at_most_one.append(all(i <= 1))
    is_at_most_one = np.array(is_at_most_one)
    assert all(is_at_most_one)
