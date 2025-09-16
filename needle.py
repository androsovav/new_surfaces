# needle.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from optics import Stack
from design import insert_layer
from merit import rms_merit

def discrete_excitation_map(
    stack: Stack,
    wavelengths: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    n_candidates: List[float],
    kind: str = "R",
    pol: str = "s",
    theta_inc: float = 0.0,
    d_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (positions, n_best, dMF) по дискретным позициям вставки.
    """
    base_mf = rms_merit(stack, wavelengths, target, sigma, kind=kind, pol=pol, theta_inc=theta_inc)
    N = len(stack.layers)
    positions = np.arange(N + 1, dtype=int)
    best_n = np.full(N + 1, np.nan, dtype=float)
    best_dmf = np.full(N + 1, np.inf, dtype=float)

    for i in range(N + 1):
        for n_new in n_candidates:
            test = insert_layer(stack, index=i, n_new=n_new, d_new=d_eps)
            mf = rms_merit(test, wavelengths, target, sigma, kind=kind, pol=pol, theta_inc=theta_inc)
            dmf = mf - base_mf
            if dmf < best_dmf[i]:
                best_dmf[i] = dmf
                best_n[i] = n_new
    return positions, best_n, best_dmf

def needle_step(
    stack: Stack,
    wavelengths: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    n_candidates: List[float],
    kind: str = "R",
    pol: str = "s",
    theta_inc: float = 0.0,
    d_init: float = 2e-3,
    d_eps: float = 2e-4,
) -> Tuple[Stack, dict]:
    """
    Делает ОДИН шаг: находит лучшую позицию/материал для 'иглы' и вставляет тонкий слой (d_init).
    """
    pos, n_best, dmf = discrete_excitation_map(
        stack, wavelengths, target, sigma,
        n_candidates=n_candidates, kind=kind, pol=pol, theta_inc=theta_inc, d_eps=d_eps
    )
    j = int(np.argmin(dmf))
    info = {
        "position": int(pos[j]),
        "n_new": float(n_best[j]),
        "d_mf_pred": float(dmf[j]),
        "base_layers": len(stack.layers)
    }
    new_stack = insert_layer(stack, index=int(pos[j]), n_new=float(n_best[j]), d_new=d_init)
    return new_stack, info
