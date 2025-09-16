# targets.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def target_AR(wavelengths: np.ndarray, R_target: float = 0.0, sigma: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    target = np.full_like(wavelengths, R_target, dtype=float)
    sig = np.full_like(wavelengths, sigma, dtype=float)
    return target, sig

def target_bandpass(wavelengths: np.ndarray, passbands: list[tuple[float,float]], sigma_pass=0.01, sigma_stop=0.01) -> Tuple[np.ndarray, np.ndarray]:
    target = np.zeros_like(wavelengths, dtype=float)
    sigma = np.full_like(wavelengths, sigma_stop, dtype=float)
    for a,b in passbands:
        mask = (wavelengths >= a) & (wavelengths <= b)
        target[mask] = 1.0
        sigma[mask] = sigma_pass
    return target, sigma
