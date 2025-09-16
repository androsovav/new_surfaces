# optics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Literal, Union
import numpy as np

Pol = Literal["s", "p", "u"]  # u = unpolarized average
NType = Union[float, complex, Callable[[float], complex]]

@dataclass
class Layer:
    n: NType          # число или функция n(λ)->complex
    d: float          # физ. толщина (м)

@dataclass
class Stack:
    layers: List[Layer]
    n_inc: NType      # n внешней среды (число/функция)
    n_sub: NType      # n подложки (число/функция)

def _n_of(nspec: NType, wl: float) -> complex:
    return complex(nspec(wl)) if callable(nspec) else complex(nspec)

def _cos_theta_in_layer(n_layer: complex, n_incident: complex, theta_incident: float) -> complex:
    if theta_incident == 0.0:
        return 1.0
    sin_ti = np.sin(theta_incident)
    sin_tj = (n_incident * sin_ti) / n_layer
    return np.sqrt(1.0 - sin_tj**2)

def _q_parameter(n: complex, cos_theta: complex, pol: Literal["s","p"]) -> complex:
    return n * cos_theta if pol == "s" else (cos_theta / n)

def _M_layer(nj: complex, dj: float, wl: float, cosj: complex, pol: Literal["s","p"]) -> np.ndarray:
    phi = 2.0 * np.pi * nj * dj * cosj / wl
    c, s = np.cos(phi), np.sin(phi)
    q = _q_parameter(nj, cosj, pol)
    return np.array([[c, 1j * s / q],
                     [1j * q * s, c]], dtype=complex)

def _M_stack(stack: Stack, wl: float, theta_inc: float, pol: Literal["s","p"]) -> Tuple[np.ndarray, complex, complex]:
    n_in  = _n_of(stack.n_inc, wl)
    n_sub = _n_of(stack.n_sub, wl)
    cos_in  = _cos_theta_in_layer(n_in,  n_in,  theta_inc)
    cos_sub = _cos_theta_in_layer(n_sub, n_in,  theta_inc)
    M = np.eye(2, dtype=complex)
    for L in stack.layers:
        nj = _n_of(L.n, wl)
        cosj = _cos_theta_in_layer(nj, n_in, theta_inc)
        M = M @ _M_layer(nj, L.d, wl, cosj, pol)
    q_in  = _q_parameter(n_in,  cos_in,  pol)
    q_sub = _q_parameter(n_sub, cos_sub, pol)
    return M, q_in, q_sub

def rt_amplitudes(stack: Stack, wl: float, theta_inc: float = 0.0, pol: Literal["s","p"] = "s") -> Tuple[complex, complex]:
    M, q_in, q_sub = _M_stack(stack, wl, theta_inc, pol)
    m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
    denom = (m11 + m12 * q_sub) * q_in + (m21 + m22 * q_sub)
    r = ((m11 + m12 * q_sub) * q_in - (m21 + m22 * q_sub)) / denom
    t =  2.0 * q_in / denom
    return r, t

def RT_single(stack: Stack, wl: float, theta_inc: float = 0.0, pol: Pol = "s") -> Tuple[float, float]:
    if pol == "u":
        # усреднение s/p
        r_s, t_s = rt_amplitudes(stack, wl, theta_inc, "s")
        r_p, t_p = rt_amplitudes(stack, wl, theta_inc, "p")
        _, q_in_s,  q_sub_s = _M_stack(stack, wl, theta_inc, "s")
        _, q_in_p,  q_sub_p = _M_stack(stack, wl, theta_inc, "p")
        R = 0.5 * (np.abs(r_s)**2 + np.abs(r_p)**2)
        T = 0.5 * (np.real(q_sub_s/q_in_s)*np.abs(t_s)**2 + np.real(q_sub_p/q_in_p)*np.abs(t_p)**2)
        return float(R), float(T)
    r, t = rt_amplitudes(stack, wl, theta_inc, pol)
    _, q_in,  q_sub = _M_stack(stack, wl, theta_inc, pol)
    R = np.abs(r)**2
    T = np.real(q_sub/q_in) * np.abs(t)**2
    return float(R), float(T)

def RT(stack: Stack, wavelengths: np.ndarray, theta_inc: float = 0.0, pol: Pol = "s") -> Tuple[np.ndarray, np.ndarray]:
    R = np.empty_like(wavelengths, dtype=float)
    T = np.empty_like(wavelengths, dtype=float)
    for i, wl in enumerate(wavelengths):
        R[i], T[i] = RT_single(stack, float(wl), theta_inc, pol)
    return R, T
