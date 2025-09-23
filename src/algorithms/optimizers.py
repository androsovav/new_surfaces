# src/algorithms/optimizers.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Literal
from ..core.optics import Stack
from ..core.merit import rms_merit
from ..design.design import make_stack


def coordinate_descent_thicknesses(
    stack: Stack,
    wavelengths: np.ndarray,
    n_wavelengths: int,
    targets: dict,
    nH_values: np.ndarray,
    nL_values: np.ndarray,
    cos_theta_in_H_layers: np.ndarray,
    cos_theta_in_L_layers: np.ndarray,
    q_in: np.ndarray,
    q_sub: np.ndarray,
    qH: np.ndarray,
    qL: np.ndarray,
    pol: Literal["s","p"] = "s",
    theta_inc: float = 0.0,
    step_rel: float = 0.1,
    min_step_rel: float = 0.01,
    iters: int = 50,
    d_min: float | None = None,
    d_max: float | None = None,
) -> Tuple[Stack, float]:
    """
    Координатный спуск по толщинам с постепенным уменьшением шага.
    """

    start_flag = stack.layers[0].litera
    thickness = np.array([L.d for L in stack.layers], dtype=float)

    # функция для пересборки стека
    def rebuild(th):
        return make_stack(
            start_flag, th, nH_values,
            nL_values, cos_theta_in_H_layers,
            cos_theta_in_L_layers, q_in,
            q_sub, qH, qL, wavelengths, n_wavelengths,
            calculate_prefix_and_suffix_for_needle=False
        )

    base = rebuild(thickness)
    mf = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc,
                   base.r, base.t, base.R, base.T)
    step = step_rel

    def clamp(d):
        if d_min is not None: d = max(d, d_min)
        if d_max is not None: d = min(d, d_max)
        return d

    for _ in range(iters):
        improved = False
        for i in range(len(thickness)):
            d0 = thickness[i]
            for sgn in (+1, -1):
                th_try = thickness.copy()
                th_try[i] = clamp(d0 * (1.0 + sgn * step))
                test = rebuild(th_try)
                mf_new = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc,
                                   test.r, test.t, test.R, test.T)
                if mf_new < mf:
                    mf = mf_new
                    thickness = th_try
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < min_step_rel:
                break

    return test, mf
