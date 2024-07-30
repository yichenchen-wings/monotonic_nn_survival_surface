from monotonic_nn_surv_surf.core.survsurf_2d_sigm import SurvSurf2DSigm
import torch
import numpy as np
import pytest

@pytest.mark.parametrize(
        "batch_size, hidden_dim, n_layers, feat_size", 
        [
            (1, 3, 2, 2), 
            (3, 2, 4, 1),
            (5, 6, 3, 6)
        ]
)
def test_SurvSurf2DSigm_shape(batch_size, hidden_dim, n_layers, feat_size):
    ts = torch.rand(batch_size, 1)
    gs = torch.rand(batch_size, 1)
    xs = torch.rand(batch_size, feat_size)

    surf = SurvSurf2DSigm(
            z0_size=feat_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers
    )
    out = surf(ts=ts, gs=gs, xs=xs)
    assert out.shape == (batch_size, 1)


@pytest.mark.parametrize(
    "seed, hidden_dim, n_layers", 
    [
        (i, 32, 2) for i in range(1, 100, 10)
    ] + [
        (i, 48, 3) for i in range(100, 200, 10)
    ] + [
        (i, 64, 4) for i in range(200, 300, 10)
    ]
)
def test_SurvSurf2DSigm_monotone(seed, hidden_dim, n_layers):
    torch.manual_seed(seed)
    batch_size = 100
    feat_size = 2
    ts = torch.rand(batch_size, 1) * 100
    gs = torch.rand(batch_size, 1) * 100
    ts.requires_grad = True
    gs.requires_grad = True

    xs = torch.rand(batch_size, feat_size)*100 - 50

    surf = SurvSurf2DSigm(
        z0_size=feat_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    out = surf(ts=ts, gs=gs, xs=xs)

    is_grad_non_neg = []
    is_grad_non_pos = []
    min_grad_non_neg = np.inf
    max_grad_non_pos = -np.inf
    for i in out:
        i.backward(retain_graph=True)
        is_grad_non_neg.append(all(ts.grad >= -1e-6))
        is_grad_non_pos.append(all(gs.grad <= 1e-6))
        min_grad_non_neg = min(min_grad_non_neg, min(ts.grad))
        max_grad_non_pos = max(max_grad_non_pos, max(gs.grad))

    is_grad_non_neg = np.array(is_grad_non_neg)
    is_grad_non_pos = np.array(is_grad_non_pos)
    assert all(is_grad_non_neg), f'min grad for the non-neg varible is {min_grad_non_neg}'
    assert all(is_grad_non_pos), f'max grad for the non-pos variable is {max_grad_non_pos}'


@pytest.mark.parametrize(
    "seed", 
    [i for i in range(1,100,5)]
)
def test_SurvSurf2DSigm_range(seed):
    torch.manual_seed(seed)
    batch_size = 30
    feat_size = 2
    ts = torch.zeros(batch_size, 1)
    gs = torch.rand(batch_size, 1) * 100
    xs = torch.rand(batch_size, feat_size) * 100 - 50

    surf = SurvSurf2DSigm(
        z0_size=feat_size,
        hidden_dim=64,
        n_layers=3
    )
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
def test_SurvSurf2DSigm_max(seed):
    torch.manual_seed(seed)
    batch_size = 30
    feat_size = 2
    ts = torch.rand(batch_size, 1) * 1000
    gs = torch.rand(batch_size, 1) * 1000
    xs = torch.rand(batch_size, feat_size)*1000 - 500

    surf = SurvSurf2DSigm(
        z0_size=feat_size,
        hidden_dim=64,
        n_layers=3
    )
    out = surf(ts=ts, gs=gs, xs=xs)

    is_at_most_one = []
    for i in out:
        is_at_most_one.append(all(i <= 1))
    is_at_most_one = np.array(is_at_most_one)
    assert all(is_at_most_one)
