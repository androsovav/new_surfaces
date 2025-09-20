# src/design/design.py
from __future__ import annotations
from typing import List, Callable, Literal
import numpy as np
from ..core.optics import NType, n_of, Layer, Stack, cos_theta_in_layer, q_parameter, phi_parameter, make_M

def make_stack(start_flag: Literal["H", "L"], thickness: np.ndarray, n_inc: NType, n_sub: NType, nH: NType, nL: NType, theta_inc: float, wl: float, pol: Literal["s","p"]) -> Stack:
    newwl = 550e-9 # заглушка
    num_of_layers = len(thickness)
    layers = np.ndarray(shape=(num_of_layers), dtype=Layer)
    Hlayer = start_flag == "H"
    for i in range(num_of_layers):
        if Hlayer:
            Hlayer = False
            litera = "H"
            n = n_of(nspec=nH, wl=newwl)
        else:
            Hlayer = True
            litera = "L"
            n = n_of(nspec=nL, wl=newwl)
        cos_theta=cos_theta_in_layer(n, n_inc, theta_inc)
        phi = phi_parameter(n, thickness[i], cos_theta, newwl)
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        q = q_parameter(n, cos_theta, pol)
        layers[i] = Layer(litera=litera,
                            n=n, 
                            d=thickness[i],
                            cos_theta=cos_theta,
                            phi=phi,
                            sphi=sphi,
                            cphi=cphi,
                            q=q,
                            M=make_M(sphi=sphi, cphi=cphi, q=q))
    prefix = np.ndarray(shape= (num_of_layers), dtype=np.ndarray)
    suffix = np.ndarray(shape= (num_of_layers), dtype=np.ndarray)
    left = np.eye(2, dtype=complex)
    right = np.eye(2, dtype=complex)
    for i in range(num_of_layers):
        # считаем матрицу половины слоя
        layer = layers[i]
        layer.d = 0.5*layer.d
        layer.phi = phi_parameter(layer.n, layer.d, layer.cos_theta, newwl)
        layer.sphi = np.sin(layer.phi)
        layer.cphi = np.cos(layer.phi)
        # домножаем произведение М всех слоев слева на М половины слоя
        prefix[i] = left @ make_M(layer.sphi, layer.cphi, layer.q)
        # добавляем к произведению новый слой
        left = left @ layers[i].M
        # считаем матрицу половины слоя
        layer = layers[-(i+1)]
        layer.d = 0.5*layer.d
        layer.phi = phi_parameter(layer.n, layer.d, layer.cos_theta, newwl)
        layer.sphi = np.sin(layer.phi)
        layer.cphi = np.cos(layer.phi)
        # домножаем произведение М всех слоев слева на М половины слоя
        suffix[i] = make_M(layer.sphi, layer.cphi, layer.q) @ right
        # добавляем к произведению новый слой
        right = layers[-(i+1)].M @ right
        # считаем матрицу половины слоя
        layer = layers[i]
        layer.d = 0.5*layer.d
        layer.phi = phi_parameter(layer.n, layer.d, layer.cos_theta, newwl)
        layer.sphi = np.sin(layer.phi)
        layer.cphi = np.cos(layer.phi)
        # домножаем произведение М всех слоев слева на М половины слоя
        prefix[i] = left @ make_M(layer.sphi, layer.cphi, layer.q)
        # добавляем к произведению новый слой
        left = left @ layers[i].M
        # считаем матрицу половины слоя
        layer = layers[-(i+1)]
        layer.d = 0.5*layer.d
        layer.phi = phi_parameter(layer.n, layer.d, layer.cos_theta, newwl)
        layer.sphi = np.sin(layer.phi)
        layer.cphi = np.cos(layer.phi)
        # домножаем M половины слоя на произведение M всех слоев справа
        suffix[i] = make_M(layer.sphi, layer.cphi, layer.q) @ right
        # добавляем к произведению новый слой
        right = layers[-(i+1)].M @ right
    m = left
    print(layers)
    print("layers")
    print(n_inc)
    print("n_inc")
    print(n_sub)
    print("n_sub")
    print(prefix)
    print("prefix")
    print(suffix)
    print("suffix")
    print(m)
    print("M")
    return Stack(layers=layers, n_inc=n_inc, n_sub=n_sub, prefix=prefix, suffix=suffix, M=m)

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
