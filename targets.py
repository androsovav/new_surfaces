# targets.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def target_AR(wavelengths: np.ndarray, R_target: float = 0.0, sigma: float = 0.01):
    return {"R": {"target": np.full_like(wavelengths, R_target, dtype=float),
                  "sigma":  np.full_like(wavelengths, sigma, dtype=float)}}

def target_bandpass(wavelengths: np.ndarray,
                    passbands: list[tuple[float,float]],
                    sigma_pass=0.01, sigma_stop=0.01):
    target = np.zeros_like(wavelengths, dtype=float)
    sigma  = np.full_like(wavelengths, sigma_stop, dtype=float)
    for a,b in passbands:
        mask = (wavelengths >= a) & (wavelengths <= b)
        target[mask] = 1.0
        sigma[mask]  = sigma_pass
    return {"T": {"target": target, "sigma": sigma}}

def target_low_reflect(wavelengths: np.ndarray,
                       bands: list[tuple[float,float]],
                       R_val=0.01, sigma=0.01):
    target = np.full_like(wavelengths, 1.0, dtype=float)
    sigmaA = np.full_like(wavelengths, 1.0, dtype=float)
    # зададим цель по отражению R≈R_val в bands
    R = np.full_like(wavelengths, R_val, dtype=float)
    S = np.full_like(wavelengths, sigma, dtype=float)
    mask = np.zeros_like(wavelengths, dtype=bool)
    for a,b in bands:
        mask |= (wavelengths >= a) & (wavelengths <= b)
    return {"R": {"target": R, "sigma": S}}

def combine_targets(*targets: dict) -> dict:
    combined = {}
    for t in targets:
        for k, v in t.items():
            if k in combined:
                # если ключ повторяется, объединим по правилу "последний перезаписывает"
                pass
            combined[k] = v
    return combined
