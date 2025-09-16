# metrics.py
from __future__ import annotations
from typing import Optional
import numpy as np
from optics import Stack, _n_of

def layer_count(stack: Stack) -> int:
    return len(stack.layers)

def total_optical_thickness(stack: Stack, wl_ref: float) -> float:
    """
    TOT = Σ n(λ_ref) * d  [метры-оптические].
    Для отчётов удобно конвертировать в нм-опт: TOT_nmopt = TOT / 1e-9
    """
    return float(sum(_n_of(L.n, wl_ref).real * L.d for L in stack.layers))
