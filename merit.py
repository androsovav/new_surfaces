# merit.py
from __future__ import annotations
import numpy as np
from typing import Literal, Sequence
from optics import Stack, RT, rt_amplitudes

# типы целей
TargetKind = Literal["R", "T", "phase_t", "phase_r"]

def _phase(z: complex) -> float:
    return np.angle(z)

def rms_merit(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict[str, dict[str, np.ndarray]],
    pol: Literal["s","p"] = "s",
    theta_inc: float = 0.0,
) -> float:
    """
    Универсальная RMS-мерит функция для многокритериальных целей.
    targets = {
       "R": {"target": ..., "sigma": ...},
       "T": {"target": ..., "sigma": ...},
       "phase_t": {"target": ..., "sigma": ...},
       "phase_r": {"target": ..., "sigma": ...},
    }
    """
    R, T = RT(stack, wavelengths, theta_inc=theta_inc, pol=pol)
    errs = []

    if "R" in targets:
        resid = (R - targets["R"]["target"]) / targets["R"]["sigma"]
        errs.append(resid**2)
    if "T" in targets:
        resid = (T - targets["T"]["target"]) / targets["T"]["sigma"]
        errs.append(resid**2)
    if "phase_t" in targets or "phase_r" in targets:
        # фазовые цели требуют амплитуд
        r, t = [], []
        for wl in wavelengths:
            ri, ti = rt_amplitudes(stack, float(wl), theta_inc, pol)
            r.append(ri); t.append(ti)
        r, t = np.array(r), np.array(t)
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
