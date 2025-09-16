# metrics.py
from __future__ import annotations
from optics import Stack, _n_of

def layer_count(stack: Stack) -> int:
    return len(stack.layers)

def total_optical_thickness(stack: Stack, wl_ref: float) -> float:
    return float(sum(_n_of(L.n, wl_ref).real * L.d for L in stack.layers))
