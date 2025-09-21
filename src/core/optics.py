# src/core/optics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Literal, Union
import numpy as np

Pol = Literal["s", "p", "u"]  # поляризация. u - неполяризованный свет.
NType = Union[float, complex, Callable[[float, float], complex]]   # показатель преломления среды. Может быть действительным (float), комплексным (complex), и задаваться функцией длины волны

@dataclass
class Layer:
    litera: Literal["H", "L"]   # тип материала (H или L)
    d: float                # физ. толщина (м)
    phi: np.ndarray         # фазовый набег (массив)
    sphi: np.ndarray        # синус фи (массив)
    cphi: np.ndarray        # косинус фи (массив)
    M: np.ndarray           # матрица слоя (3D массив: [2, 2, n_wavelengths])

@dataclass
class Stack:
    layers: np.ndarray
    prefix: np.ndarray     # префиксное произведение (3D массив)
    suffix: np.ndarray     # суффиксное произведение (3D массив)
    M: np.ndarray     # M всего стэка (3D массив)
    r: np.ndarray
    t: np.ndarray
    R: np.ndarray
    T: np.ndarray

def n_of(nspec: NType, A: float, wl: float) -> complex:
    """
    Функция, которая принимает на вход NType и возвращает одно комплексное значение показателя преломления среды
    """
    return complex(nspec(A, wl)) if callable(nspec) else complex(nspec)

def n_cauchy(A:float, wl: float) -> complex:
    # wl в метрах → переведём в мкм для удобства
    wl_um = wl * 1e6
    B, C = 0.004, 0.0001  # коэффициенты
    return A + B / wl_um**2 + C / wl_um**4

def cos_theta_in_layer(n_layer: complex, n_incident: complex, theta_incident: float) -> complex:
    """
    Функция расчета косинуса угла распространения света в слое по закону Снеллиуса
    """
    if theta_incident == 0.0:
        return 1.0
    sin_ti = np.sin(theta_incident)
    sin_tj = (n_incident * sin_ti) / n_layer
    return np.sqrt(1.0 - sin_tj**2)

# В optics.py добавьте векторные версии функций
def phi_parameter(n: np.ndarray, d: float, cos_theta: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Векторная версия phi_parameter"""
    return 2.0 * np.pi * n * d * cos_theta / wavelengths

def q_parameter(n: np.ndarray, cos_theta: np.ndarray, pol: Literal["s","p"]) -> np.ndarray:
    """Векторная версия q_parameter"""
    return n * cos_theta if pol == "s" else (cos_theta / n)

def make_M(sphi: np.ndarray, cphi: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
    """Векторная версия make_M"""
    M = np.empty((2, 2, n), dtype=complex)
    M[0, 0] = cphi
    M[0, 1] = 1j * sphi / q
    M[1, 0] = 1j * q * sphi
    M[1, 1] = cphi
    return M

def rt_amplitudes(M: np.ndarray, q_in: np.ndarray, q_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет амплитуды r, t для всего стека.
    M – матрицы стека (2,2,nλ)
    q_in, q_sub – параметры среды и подложки (nλ,)
    """
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    X = A + B*q_sub
    Y = C + D*q_sub
    denom = X*q_in + Y
    r = (X*q_in - Y) / denom
    t = (2*q_in) / denom
    return r, t

def RT_coeffs(r: np.ndarray, t: np.ndarray, q_in: np.ndarray, q_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет коэффициенты отражения и пропускания (энергетические).
    """
    R = np.abs(r)**2
    T = (np.real(q_sub) / np.real(q_in)) * np.abs(t)**2
    return R, T


def RT_single(stack: Stack, q_in, q_sub, wl: float, theta_inc: float, pol: Literal["s","p"]) -> Tuple[float, float]:
    r, t = rt_amplitudes(stack, q_in, q_sub, wl, theta_inc, pol)
    R = np.abs(r)**2
    T = np.real(q_sub/q_in) * np.abs(t)**2
    return float(R), float(T)

def RT(stack: Stack, q_in, q_sub, wavelengths: np.ndarray, theta_inc: float, pol: Literal["s","p"]) -> Tuple[np.ndarray, np.ndarray]:
    R = np.empty_like(wavelengths, dtype=float)
    T = np.empty_like(wavelengths, dtype=float)
    for i, wl in enumerate(wavelengths):
        R[i], T[i] = RT_single(stack, q_in, q_sub, float(wl), theta_inc, pol)
    return R, T
