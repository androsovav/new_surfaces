# src/design/design.py
from __future__ import annotations
from typing import List, Callable, Literal
import numpy as np
from ..core.optics import NType, n_of, Layer, Stack, cos_theta_in_layer, q_parameter, phi_parameter, make_M

def make_stack(start_flag: Literal["H", "L"], thickness: np.ndarray, 
               n_inc_values: np.ndarray, n_sub_values: np.ndarray, nH_values: np.ndarray, nL_values: np.ndarray, 
               cos_theta_in_H_layers: np.ndarray, cos_theta_in_L_layers: np.ndarray, qH: complex,
                qL: complex, wavelengths: np.ndarray, n_wavelengths: int, pol: Literal["s","p"]) -> Stack:
    
    num_of_layers = len(thickness)
    
    # Инициализация массивов для всех длин волн
    layers = np.empty(num_of_layers, dtype=object)
    Hlayer = start_flag == "H"

    for i in range(num_of_layers):
        if Hlayer:
            Hlayer = False
            litera = "H"
            # Вычисляем phi, sphi, cphi для всех длин волн
            phi = phi_parameter(nH_values, thickness[i], cos_theta_in_H_layers, wavelengths)
            sphi = np.sin(phi)
            cphi = np.cos(phi)
            # Вычисляем матрицы M для всех длин волн
            M = make_M(sphi, cphi, qH, n_wavelengths)
            
            layers[i] = Layer(litera=litera, d=thickness[i], phi=phi, sphi=sphi, cphi=cphi, M=M)
        else:
            Hlayer = True
            litera = "L"
            # Вычисляем phi, sphi, cphi для всех длин волн
            phi = phi_parameter(nL_values, thickness[i], cos_theta_in_L_layers, wavelengths)
            sphi = np.sin(phi)
            cphi = np.cos(phi)
            # Вычисляем матрицы M для всех длин волн
            M = make_M(sphi, cphi, qL, n_wavelengths)
            
            layers[i] = Layer(litera=litera, d=thickness[i], phi=phi, sphi=sphi, cphi=cphi, M=M)
            
    # Инициализация префиксных и суффиксных произведений
    prefix = np.empty((num_of_layers, 2, 2, n_wavelengths), dtype=complex)
    suffix = np.empty((num_of_layers, 2, 2, n_wavelengths), dtype=complex)
    
    left = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, n_wavelengths))
    right = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, n_wavelengths))
    
    for i in range(num_of_layers):
        # Обработка префикса
        layer = layers[i]
        half_d = 0.5 * layer.d
        
        # Пересчитываем параметры для половины толщины
        if layer.litera == "H":
            phi_half = phi_parameter(nH_values, half_d, cos_theta_in_H_layers, wavelengths)
            sphi_half = np.sin(phi_half)
            cphi_half = np.cos(phi_half)
            # Создаем матрицу для половины слоя
            M_half = make_M(sphi_half, cphi_half, qH, n_wavelengths)
        else:
            phi_half = phi_parameter(nL_values, half_d, cos_theta_in_L_layers, wavelengths)
            sphi_half = np.sin(phi_half)
            cphi_half = np.cos(phi_half)            
            # Создаем матрицу для половины слоя
            M_half = make_M(sphi_half, cphi_half, qL, n_wavelengths)        

        
        # Обновляем префикс и left
        prefix[i] = np.einsum('ijk,jlk->ilk', left, M_half)
        left = np.einsum('ijk,jlk->ilk', left, layer.M)
        
        # Обработка суффикса (аналогично для обратного порядка)
        layer_rev = layers[-(i+1)]
        half_d_rev = 0.5 * layer_rev.d
        
        if layer_rev.litera == "H":
            phi_half_rev = phi_parameter(nH_values, half_d_rev, cos_theta_in_H_layers, wavelengths)
            sphi_half_rev = np.sin(phi_half_rev)
            cphi_half_rev = np.cos(phi_half_rev)     
            M_half_rev = make_M(sphi_half_rev, cphi_half_rev, qH, n_wavelengths)
        else:
            phi_half_rev = phi_parameter(nL_values, half_d_rev, cos_theta_in_L_layers, wavelengths)
            sphi_half_rev = np.sin(phi_half_rev)
            cphi_half_rev = np.cos(phi_half_rev)     
            M_half_rev = make_M(sphi_half_rev, cphi_half_rev, qL, n_wavelengths)
        
        suffix[-(i+1)] = np.einsum('ijk,jlk->ilk', M_half_rev, right)
        right = np.einsum('ijk,jlk->ilk', layer_rev.M, right)
    
    return Stack(layers=layers, prefix=prefix, suffix=suffix, M=left)

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
