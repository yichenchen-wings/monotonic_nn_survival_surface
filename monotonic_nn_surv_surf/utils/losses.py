
import torch
import numpy as np
def loss_dydg(model, ts, gs, xs, ys, weights):
    with torch.inference_mode(False):
        gs.requires_grad_()
        model.eval()
        outputs = model(ts=ts, gs=gs, xs=xs)
        dydg = torch.autograd.grad(
            outputs=outputs,
            inputs=gs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]
    assert torch.all(1e-2 > dydg), f"{torch.max(dydg)}"
    dydg = torch.clamp(dydg, -np.inf, 0)
    epsilon = 1e-6
    #dydg_rescale = torch.exp(dydg)
    #minus_dydg_rescale = 1 - dydg_rescale 
    losses = (
        #ys*torch.log(outputs + epsilon) # if observed g (g > 0) at t, then (t,g,x) should have prob closer to 1
        + ys*torch.log(-dydg + epsilon) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
        + (1-ys)*torch.log(1-outputs + epsilon) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
        #+ (1-ys)*torch.log(dydg_rescale + epsilon) # if g = 0 at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g_min/2,x) should be low.
    ) 
    
    losses = -torch.mean(weights*losses)
    return losses

class LossDyAcrossGResol:
    def __init__(self, g_resol):
        self.g_resol = g_resol
    def __call__(self, model, ts, gs, xs, ys, weights):
        model.eval()
        outputs = model(ts=ts, gs=gs, xs=xs)
        outputs_bigger_g = model(ts=ts, gs=gs+self.g_resol, xs=xs)
        dy = outputs_bigger_g - outputs #lager in magnitude the better if y = 1
        assert torch.all(1e-2 > dy), f"{torch.max(dy)}"
        dy = torch.clamp(dy, -np.inf, 0)
        epsilon = 1e-6
        losses = (
            ys*(torch.log(-dy + epsilon) + torch.log(1-outputs_bigger_g + epsilon))# considered only if observed g_obs > 0 at t
            + (1-ys)*torch.log(1-outputs + epsilon) # considered only if observed g_obs == 0 at t
        ) 
        
        losses = -torch.mean(weights*losses)
        return losses

class BCEDyAcrossGResol:
    def __init__(self, g_resol):
        self.g_resol = g_resol
    def __call__(self, model, ts, gs, xs, ys, weights):
        model.eval()
        outputs = model(ts=ts, gs=gs, xs=xs)
        outputs_bigger_g = model(ts=ts, gs=gs+self.g_resol, xs=xs)
        dy = outputs_bigger_g - outputs #lager in magnitude the better if y = 1
        assert torch.all(1e-2 > dy), f"{torch.max(dy)}"
        dy = torch.clamp(dy, -np.inf, 0)
        epsilon = 1e-6
        losses = (
            ys*(torch.log(outputs + epsilon) + torch.log(1-outputs_bigger_g + epsilon))# considered only if observed g_obs > 0 at t
            + (1-ys)*torch.log(1-outputs + epsilon) # considered only if observed g_obs == 0 at t
        ) 
        
        losses = -torch.mean(weights*losses)
        return losses

def loss_bce(model, ts, gs, xs, ys, weights):
    outputs = model(ts=ts, gs=gs, xs=xs)
    loss_fn = torch.nn.functional.binary_cross_entropy
    epsilon = torch.finfo(outputs.dtype).tiny

    losses = loss_fn(outputs + epsilon, ys, weight=weights)
    return losses