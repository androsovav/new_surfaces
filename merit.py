# merit.py
from __future__ import annotations
import numpy as np
from typing import Literal
from optics import Stack, RT

TargetKind = Literal["R", "T"]

def rms_merit(
    stack: Stack,
    wavelengths: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    kind: TargetKind = "R",
    pol: Literal["s","p"] = "s",
    theta_inc: float = 0.0,
) -> float:
    """
    RMS-ошибка между рассчитанным спектром (R или T) и целевым, с весами sigma.
    """
    R, T = RT(stack, wavelengths, theta_inc=theta_inc, pol=pol)
    model = R if kind == "R" else T
    resid = (model - target) / sigma
    return float(np.sqrt(np.mean(resid**2)))
