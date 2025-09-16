# optimizers.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from optics import Stack
from merit import rms_merit

def coordinate_descent_thicknesses(
    stack: Stack,
    wavelengths: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    kind: str = "R",
    pol: str = "s",
    theta_inc: float = 0.0,
    step_rel: float = 0.1,
    min_step_rel: float = 0.01,
    iters: int = 50,
    d_min: float | None = None,
    d_max: float | None = None,
) -> Tuple[Stack, float]:
    """
    Простейшая локальная оптимизация толщин: координатный спуск с уменьшением шага.
    """
    layers = [l for l in stack.layers]
    base = Stack(layers=layers, n_inc=stack.n_inc, n_sub=stack.n_sub)
    mf = rms_merit(base, wavelengths, target, sigma, kind=kind, pol=pol, theta_inc=theta_inc)
    step = step_rel

    def clamp(d):
        if d_min is not None: d = max(d, d_min)
        if d_max is not None: d = min(d, d_max)
        return d

    for _ in range(iters):
        improved = False
        for i in range(len(layers)):
            d0 = layers[i].d
            for sgn in (+1, -1):
                layers[i].d = clamp(d0 * (1.0 + sgn * step))
                test = Stack(layers=layers, n_inc=stack.n_inc, n_sub=stack.n_sub)
                mf_new = rms_merit(test, wavelengths, target, sigma, kind=kind, pol=pol, theta_inc=theta_inc)
                if mf_new < mf:
                    mf = mf_new
                    improved = True
                    d0 = layers[i].d
                else:
                    layers[i].d = d0
        if not improved:
            step *= 0.5
            if step < min_step_rel:
                break
    return Stack(layers=layers, n_inc=stack.n_inc, n_sub=stack.n_sub), mf
