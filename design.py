# design.py
from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass
from optics import Layer, Stack

def make_stack(n_inc: float, n_sub: float, nH: float, nL: float, periods: int, quarter_at: float) -> Stack:
    """
    Простейший старт: (HL)^periods с четвертьволновыми слоями на λ0=quarter_at при нормальном падении.
    """
    dH = (quarter_at / (4.0 * nH))
    dL = (quarter_at / (4.0 * nL))
    layers: List[Layer] = []
    for _ in range(periods):
        layers.append(Layer(n=nH, d=dH))
        layers.append(Layer(n=nL, d=dL))
    return Stack(layers=layers, n_inc=n_inc, n_sub=n_sub)

def with_dispersion(n_func_H: Callable[[float], complex], n_func_L: Callable[[float], complex],
                    dH: float, dL: float, periods: int, n_inc: float, n_sub: float) -> Stack:
    layers = []
    for _ in range(periods):
        layers.append(Layer(n=n_func_H, d=dH))
        layers.append(Layer(n=n_func_L, d=dL))
    return Stack(layers=layers, n_inc=n_inc, n_sub=n_sub)

def insert_layer(stack: Stack, index: int, n_new: float, d_new: float) -> Stack:
    """Вставить новый слой ПЕРЕД существующим с индексом index (0..len)."""
    new_layers = list(stack.layers)
    new_layers.insert(index, Layer(n=n_new, d=d_new))
    return Stack(layers=new_layers, n_inc=stack.n_inc, n_sub=stack.n_sub)
