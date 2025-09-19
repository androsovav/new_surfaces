# src/design/design.py
from __future__ import annotations
from typing import List, Callable, Literal
import numpy as np
from ..core.optics import NType, n_of, Layer, Stack, cos_theta_in_layer, q_parameter, make_M

def make_stack(n_inc: NType, n_sub: NType, nH: NType, nL: NType, theta_inc: float, wl: float, pol: Literal["s","p"], periods: int, quarter_at: float) -> Stack:
    dH = (quarter_at / (4.0 * nH))
    dL = (quarter_at / (4.0 * nL))
    layers: List[Layer] = []
    for _ in range(periods):
        n = n_of(nspec=nH, wl=wl)
        cos_theta=np.cos(cos_theta_in_layer(n_layer=nH, n_incident=n_inc, theta_incident=theta_inc))
        phi=2.0 * np.pi * n * dH * cos_theta / wl
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        q = q_parameter(n=n, cos_theta=cos_theta, pol=pol)
        layers.append(Layer(n=n, 
                            d=dH,
                            cos_theta=cos_theta,
                            phi=phi,
                            sphi=sphi,
                            cphi=cphi,
                            q=q,
                            M=make_M(sphi=sphi, cphi=cphi, q=q)))
        n = n_of(nspec=nL, wl=wl)
        cos_theta=np.cos(cos_theta_in_layer(n_layer=nL, n_incident=n_inc, theta_incident=theta_inc))
        phi=2.0 * np.pi * n * dL * cos_theta / wl
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        q = q_parameter(n=n, cos_theta=cos_theta, pol=pol)
        layers.append(Layer(n=n, 
                            d=dL,
                            cos_theta=cos_theta,
                            phi=phi,
                            sphi=sphi,
                            cphi=cphi,
                            q=q,
                            M=make_M(sphi=sphi, cphi=cphi, q=q)))
    prefix = np.array()
    suffix = np.array()
    return Stack(layers=layers, n_inc=n_inc, n_sub=n_sub)

def with_dispersion(n_func_H: Callable[[float], complex], n_func_L: Callable[[float], complex],
                    dH: float, dL: float, periods: int, n_inc: float, n_sub: float) -> Stack:
    layers: List[Layer] = []
    for _ in range(periods):
        layers.append(Layer(n=n_func_H, d=dH))
        layers.append(Layer(n=n_func_L, d=dL))
    return Stack(layers=layers, n_inc=n_inc, n_sub=n_sub)

def insert_layer(stack: Stack, index: int, n_new: float, d_new: float) -> Stack:
    new_layers = list(stack.layers)
    new_layers.insert(index, Layer(n=n_new, d=d_new))
    return Stack(layers=new_layers, n_inc=stack.n_inc, n_sub=stack.n_sub)

def insert_with_split(stack: Stack, layer_index: int, n_new: float, d_new: float, split_ratio: float = 0.5) -> Stack:
    assert 0.0 < split_ratio < 1.0
    L = stack.layers[layer_index]
    d1 = L.d * split_ratio
    d2 = L.d - d1
    new_layers = list(stack.layers)
    new_layers[layer_index:layer_index+1] = [Layer(n=L.n, d=d1), Layer(n=n_new, d=d_new), Layer(n=L.n, d=d2)]
    return Stack(layers=new_layers, n_inc=stack.n_inc, n_sub=stack.n_sub)

def random_start_stack(n_inc: float, n_sub: float, nH: float, nL: float,
                       wl0: float, N_layers: int, d_min: float, d_max: float,
                       rng: np.random.Generator | None = None) -> Stack:
    """
    Случайный старт: случайная последовательность H/L и толщины в [d_min, d_max].
    """
    rng = np.random.default_rng() if rng is None else rng
    layers: List[Layer] = []
    for _ in range(N_layers):
        n = nH if rng.random() < 0.5 else nL
        d = float(rng.uniform(d_min, d_max))
        layers.append(Layer(n=n, d=d))
    return Stack(layers=layers, n_inc=n_inc, n_sub=n_sub)
