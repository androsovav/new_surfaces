# src/algorithms/evolution.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple
from ..core.optics import Stack, Layer
from ..core.metrics import total_optical_thickness
from .needle import needle_cycle
from ..design.design import random_start_stack


def _scale_all_layers(stack: Stack, scale: float,
                      d_min: float | None, d_max: float | None) -> Stack:
    new = []
    for L in stack.layers:
        d = L.d * scale
        if d_min is not None:
            d = max(d, d_min)
        if d_max is not None:
            d = min(d, d_max)
        new.append(Layer(n=L.n, d=d))
    return Stack(layers=new, n_inc=stack.n_inc, n_sub=stack.n_sub)


def gradual_evolution(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: list[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    growth_factors: list[float] = [1.10, 1.10, 1.10],
    wl_ref_for_tot: float = 550e-9,
    d_min: float | None = 0.5e-9,
    d_max: float | None = None,
    **needle_kwargs,
) -> Tuple[Stack, Dict[str, Any]]:
    """
    Gradual evolution: после каждого увеличения толщин (масштабирование TOT) запускаем needle_cycle.
    """
    best_stack = stack
    best_hist: List[Dict[str, Any]] = []
    best_mf = float("inf")

    current, info = needle_cycle(stack, wavelengths, targets, n_candidates,
                                 pol=pol, theta_inc=theta_inc,
                                 d_min=d_min, d_max=d_max,
                                 wl_ref_for_tot=wl_ref_for_tot,
                                 **needle_kwargs)
    best_stack, best_hist = current, info["history"]
    best_mf = best_hist[-1].get("MF", best_hist[0]["MF"])

    for g in growth_factors:
        current = _scale_all_layers(current, g, d_min, d_max)
        current, info = needle_cycle(current, wavelengths, targets, n_candidates,
                                     pol=pol, theta_inc=theta_inc,
                                     d_min=d_min, d_max=d_max,
                                     wl_ref_for_tot=wl_ref_for_tot,
                                     **needle_kwargs)
        mf = info["history"][-1].get("MF", info["history"][0]["MF"])
        if mf < best_mf:
            best_mf = mf
            best_stack = current
            best_hist = info["history"]

    return best_stack, {"history": best_hist}


def sequential_evolution(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: list[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    steps: int = 3,
    step_growth: float = 1.05,
    wl_ref_for_tot: float = 550e-9,
    d_min: float | None = 0.5e-9,
    d_max: float | None = None,
    **needle_kwargs,
) -> Tuple[Stack, Dict[str, Any]]:
    """
    Sequential evolution: чередуем «увеличение TOT» → «needle_cycle».
    """
    history_all: List[Dict[str, Any]] = []
    current = stack
    for s in range(steps):
        current = _scale_all_layers(current, step_growth, d_min, d_max)
        current, info = needle_cycle(current, wavelengths, targets, n_candidates,
                                     pol=pol, theta_inc=theta_inc,
                                     d_min=d_min, d_max=d_max,
                                     wl_ref_for_tot=wl_ref_for_tot,
                                     **needle_kwargs)
        history_all.extend(info["history"])
    return current, {"history": history_all}


def random_starts_search(
    starts: int,
    n_inc: float, n_sub: float, nH: float, nL: float,
    wl0: float, N_layers_range: tuple[int, int],
    d_min: float, d_max: float,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: list[float],
    pol: str = "s", theta_inc: float = 0.0,
    evolution: str = "sequential",   # "sequential" | "gradual" | "none"
    evolution_kwargs: dict | None = None,
    needle_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[Stack, Dict[str, Any]]:
    """
    Несколько случайных стартов → эволюция (выбранного типа) → лучшее решение по MF.
    """
    rng = np.random.default_rng() if rng is None else rng
    evolution_kwargs = evolution_kwargs or {}
    needle_kwargs = needle_kwargs or {}

    evolution_kwargs.pop("wl_ref_for_tot", None)  # избегаем дубликата

    best_stack: Stack | None = None
    best_hist: List[Dict[str, Any]] = []
    best_mf = float("inf")

    for _ in range(starts):
        N_random = rng.integers(N_layers_range[0], N_layers_range[1] + 1)
        stack0 = random_start_stack(n_inc, n_sub, nH, nL,
                                    wl0, int(N_random), d_min, d_max, rng=rng)

        if evolution == "sequential":
            st, info = sequential_evolution(stack0, wavelengths, targets, n_candidates,
                                            pol=pol, theta_inc=theta_inc,
                                            d_min=d_min, d_max=d_max,
                                            **evolution_kwargs, **needle_kwargs)
        elif evolution == "gradual":
            st, info = gradual_evolution(stack0, wavelengths, targets, n_candidates,
                                         pol=pol, theta_inc=theta_inc,
                                         d_min=d_min, d_max=d_max,
                                         **evolution_kwargs, **needle_kwargs)
        else:
            st, info = needle_cycle(stack0, wavelengths, targets, n_candidates,
                                    pol=pol, theta_inc=theta_inc,
                                    d_min=d_min, d_max=d_max,
                                    **needle_kwargs)

        mf = info["history"][-1].get("MF", info["history"][0]["MF"])
        if mf < best_mf:
            best_mf = mf
            best_stack = st
            best_hist = info["history"]

    assert best_stack is not None
    return best_stack, {"history": best_hist}
