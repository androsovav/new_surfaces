# src/design/checks.py
from __future__ import annotations
import numpy as np
from ..core.optics import RT, Stack

def energy_check(stack: Stack, wavelengths: np.ndarray, theta_inc: float = 0.0, pol: str = "u") -> float:
    """
    Возвращает max|1 - (R+T)| по сетке λ (аппроксимация A).
    Для безабсорбционных задач должен быть ~ 1e-12 ... 1e-6.
    """
    R, T = RT(stack, wavelengths, theta_inc=theta_inc, pol=pol)
    return float(np.max(np.abs(1.0 - (R + T))))
