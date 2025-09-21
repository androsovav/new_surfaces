# src/core/merit.py
from __future__ import annotations
import numpy as np
from typing import Literal
from .optics import Stack, RT, rt_amplitudes

TargetKind = Literal["R", "T", "phase_t", "phase_r"]

def _phase(z: np.ndarray | complex) -> np.ndarray:
    return np.angle(z)

def rms_merit(
    stack: Stack,
    q_in: np.ndarray, q_sub,
    wavelengths: np.ndarray,
    targets: dict[str, dict[str, np.ndarray]],
    pol: Literal["s","p"],
    theta_inc: np.ndarray
) -> float:
    """
    Универсальная RMS-мерит функция для многокритериальных целей.
    """
    errs = []

    if "R" in targets:
        R = stack.R
        resid = (R - targets["R"]["target"]) / targets["R"]["sigma"]
        errs.append(resid**2)
    if "T" in targets:
        T = stack.T
        resid = (T - targets["T"]["target"]) / targets["T"]["sigma"]
        errs.append(resid**2)

    if "phase_t" in targets or "phase_r" in targets:
        if pol == "u":
            raise ValueError("Фазовые цели нельзя задавать при pol='u'; выберите 's' или 'p'.")
        r, t = stack.r, stack.t
        if "phase_t" in targets:
            resid = (_phase(t) - targets["phase_t"]["target"]) / targets["phase_t"]["sigma"]
            errs.append(resid**2)
        if "phase_r" in targets:
            resid = (_phase(r) - targets["phase_r"]["target"]) / targets["phase_r"]["sigma"]
            errs.append(resid**2)

    if not errs:
        return 0.0
    resid_all = np.concatenate(errs, axis=0)
    return float(np.sqrt(np.mean(resid_all)))
