# src/core/optics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Literal, Union
import numpy as np

Pol = Literal["s", "p", "u"]  # поляризация. u - неполяризованный свет.
NType = Union[float, complex, Callable[[float, float], complex]]   # показатель преломления среды. Может быть действительным (float), комплексным (complex), и задаваться функцией длины волны

@dataclass
class Stack:
    thickness: np.ndarray
    start_flag: Literal["H", "L"]
    prefix: dict[Pol, np.ndarray]     # префиксное произведение (3D массив)
    suffix: dict[Pol, np.ndarray]     # суффиксное произведение (3D массив)
    M: dict[Pol, np.ndarray]       # M всего стэка (3D массив)
    phi: np.ndarray         # фазовый набег слоев (массив)
    sphi: np.ndarray        # синус фи слоев (массив)
    cphi: np.ndarray        # косинус фи слоев (массив)
    M_layers: dict[Pol, np.ndarray]       # M слоев, то есть массив матриц
    r: dict[Pol, np.ndarray]
    t: dict[Pol, np.ndarray]
    R: dict[Pol, np.ndarray]
    T: dict[Pol, np.ndarray]
    q: dict[Pol, np.ndarray]
    
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

def cos_theta_in_layer(n_layer: complex, constants: dict) -> complex:
    """
    Функция расчета косинуса угла распространения света в слое по закону Снеллиуса
    """
    if constants["theta_inc"] == 0.0:
        return 1.0
    sin_ti = np.sin(constants["theta_inc"])
    sin_tj = (constants["n_inc"] * sin_ti) / n_layer
    return np.sqrt(1.0 - sin_tj**2)

# В optics.py добавьте векторные версии функций
def phi_parameter(n: np.ndarray, d: np.ndarray, cos_theta: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Векторная версия phi_parameter"""
    return (2.0 * np.pi * n * cos_theta / wavelengths) * d[:, None]

def q_parameter(n: np.ndarray, cos_theta: np.ndarray, constants: dict) -> np.ndarray:
    """Векторная версия q_parameter"""
    return n * cos_theta if constants["pol"] == "s" else (cos_theta / n)

def make_M(sphi: np.ndarray, cphi: np.ndarray, q: np.ndarray, num_of_layers: int, num_of_wavelengths: int) -> np.ndarray:
    """Векторная версия make_M"""
    M = np.empty((2, 2, num_of_layers, num_of_wavelengths), dtype=complex)
    M[0, 0] = cphi
    M[0, 1] = 1j * sphi / q
    M[1, 0] = 1j * q * sphi
    M[1, 1] = cphi
    return M

def rt_amplitudes(constants: dict, M: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Вычисляет амплитуды r, t для каждой поляризации.
    Всегда возвращает словарь {'s': r_s, 'p': r_p}.
    """
    r, t = {}, {}

    for pol, M_pol in M.items():
        A, B, C, D = M_pol[0,0], M_pol[0,1], M_pol[1,0], M_pol[1,1]
        q_in  = constants["q_in"][pol]
        q_sub = constants["q_sub"][pol]
        X = A + B * q_sub
        Y = C + D * q_sub
        denom = X * q_in + Y
        r[pol] = (X * q_in - Y) / denom
        t[pol] = (2 * q_in) / denom

    return {"r": r, "t": t}



def RT_coeffs(constants: dict,
              r: dict[str, np.ndarray],
              t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Энергетические коэффициенты для всех поляризаций.
    """
    R, T = {}, {}
    for pol in r.keys():
        q_in  = constants["q_in"][pol]
        q_sub = constants["q_sub"][pol]
        R[pol] = np.abs(r[pol])**2
        T[pol] = (np.real(q_sub) / np.real(q_in)) * np.abs(t[pol])**2
    return {"R": R, "T": T}


def dM_layer_dd_at_zero(q: np.ndarray, k: np.ndarray, constants: dict) -> np.ndarray:
    """
    ∂M_layer/∂d |_{d=0} для всех λ: форма (2,2,K).
    Здесь k = 2π n cosθ / λ; при d→0: cosφ≈1, sinφ≈φ=k d → dM/dd:
      d/d(d) [ [cosφ, i sinφ / q], [i q sinφ, cosφ] ]_{d=0}
      = [ [0, i k / q], [i q k, 0] ].
    """
    K = len(constants["wavelengths"])
    dM = np.zeros((2, 2, K), dtype=complex)
    dM[0, 1, :] = 1j * k / q
    dM[1, 0, :] = 1j * q * k
    return dM
