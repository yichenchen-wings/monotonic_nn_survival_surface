from monotonic_nn_surv_surf.core.monotonic_net import MonotonicNet
from monotonic_nn_surv_surf.core.survival_surface_nn import SurvivalSurface
import torch
import numpy as np
import pytest

@pytest.mark.parametrize(
        "batch_size, mid_layer_sizes, feat_size, output_size", 
        [
            (1, [2,3,4], 2, 2), 
            (3, [1,2], 4, 1),
            (5, [6,6,6], 3, 6)
        ]
)
def test_SurvivalSurface_shape(batch_size, mid_layer_sizes, feat_size, output_size):
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    xs = torch.rand(batch_size, feat_size)
    net = MonotonicNet([feat_size] + mid_layer_sizes + [output_size])

    surf = SurvivalSurface(net=net)
    out = surf(ts=ts, gs=gs, xs=xs)
    assert out.shape == (batch_size, output_size)


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
def test_SurvivalSurface_monotone(seed, mid_layer_sizes):
    torch.manual_seed(seed)
    batch_size = 100
    feat_size = 2
    ts = torch.rand(batch_size, 1) * 100
    gs = torch.rand(batch_size, 1) * 100
    ts.requires_grad = True
    gs.requires_grad = True

    xs = torch.rand(batch_size, feat_size)*100 - 50
    net = MonotonicNet([feat_size] + mid_layer_sizes + [1])

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


@pytest.mark.parametrize(
    "seed", 
    [i for i in range(1,100,5)]
)
def test_SurvivalSurface_range(seed):
    torch.manual_seed(seed)
    batch_size = 30
    feat_size = 2
    ts = torch.zeros(batch_size, 1)
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


@pytest.mark.parametrize(
    "seed", 
    [i for i in range(1,100,5)]
)
def test_SurvivalSurface_max(seed):
    torch.manual_seed(seed)
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
