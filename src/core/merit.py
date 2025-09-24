# src/core/merit.py
from __future__ import annotations
import numpy as np
from typing import Literal

TargetKind = Literal["R", "T", "phase_t", "phase_r"]

def _phase(z: np.ndarray | complex) -> np.ndarray:
    return np.angle(z)

def rms_merit(
    q_in: np.ndarray, q_sub,
    wavelengths: np.ndarray,
    targets: dict[str, dict[str, np.ndarray]],
    pol: Literal["s","p"],
    theta_inc: np.ndarray,
    r: np.ndarray = np.array([]),
    t: np.ndarray = np.array([]),
    R: np.ndarray = np.array([]),
    T: np.ndarray = np.array([])
) -> float:
    """
    Универсальная RMS-мерит функция для многокритериальных целей.
    """
    errs = []

    if "R" in targets:
        resid = (R - targets["R"]["target"]) / targets["R"]["sigma"]
        errs.append(resid**2)
    if "T" in targets:
        resid = (T - targets["T"]["target"]) / targets["T"]["sigma"]
        errs.append(resid**2)

    if "phase_t" in targets or "phase_r" in targets:
        if pol == "u":
            raise ValueError("Фазовые цели нельзя задавать при pol='u'; выберите 's' или 'p'.")
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

def rms_merit_layers(
    q_in: np.ndarray, q_sub: np.ndarray,
    wavelengths: np.ndarray,
    targets: dict[str, dict[str, np.ndarray]],
    pol: str,
    theta_inc: float,
    r: np.ndarray,
    t: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> np.ndarray:
    """
    RMS-мерит функция для массива решений.
    r, t, R, T имеют форму (num_layers, n_wavelengths).
    Возвращает массив длины num_layers.
    """
    errs_all = []

    if "R" in targets:
        resid = (R - targets["R"]["target"][None, :]) / targets["R"]["sigma"][None, :]
        errs_all.append(resid**2)
    if "T" in targets:
        resid = (T - targets["T"]["target"][None, :]) / targets["T"]["sigma"][None, :]
        errs_all.append(resid**2)

    if "phase_t" in targets or "phase_r" in targets:
        if pol == "u":
            raise ValueError("Фазовые цели нельзя задавать при pol='u'.")
        if "phase_t" in targets:
            resid = (np.angle(t) - targets["phase_t"]["target"][None, :]) / targets["phase_t"]["sigma"][None, :]
            errs_all.append(resid**2)
        if "phase_r" in targets:
            resid = (np.angle(r) - targets["phase_r"]["target"][None, :]) / targets["phase_r"]["sigma"][None, :]
            errs_all.append(resid**2)

    if not errs_all:
        return np.zeros(r.shape[0])


    resid_all = np.concatenate(errs_all, axis=1)  # (num_layers, N_total)
    return np.sqrt(np.mean(resid_all, axis=1))    # (num_layers,)
