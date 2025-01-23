# monotonic_nn_survival_surface

Code for modeling a survival surface $F(g, t, x)$ with a neural network that is monotonic to $g$ and $t$ but not $x$.

`/monotonic_nn_surv_surf/core/survsurf_2d_sigm.py` contains the published model (`SurvSurf2DTaddTG`). The others are suboptimal variants.