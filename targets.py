# targets.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def target_AR(wavelengths: np.ndarray, R_target: float = 0.0, sigma: float = 0.01):
    return {"R": {"target": np.full_like(wavelengths, R_target), "sigma": np.full_like(wavelengths, sigma)}}

def target_bandpass(wavelengths: np.ndarray, passbands: list[tuple[float,float]], sigma_pass=0.01, sigma_stop=0.01):
    target = np.zeros_like(wavelengths, dtype=float)
    sigma = np.full_like(wavelengths, sigma_stop, dtype=float)
    for a,b in passbands:
        mask = (wavelengths >= a) & (wavelengths <= b)
        target[mask] = 1.0
        sigma[mask] = sigma_pass
    return {"T": {"target": target, "sigma": sigma}}

def combine_targets(*targets: dict) -> dict:
    """
    Объединяет несколько целей в один словарь для rms_merit.
    """
    combined = {}
    for t in targets:
        combined.update(t)
    return combined
