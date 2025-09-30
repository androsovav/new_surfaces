# src/core/merit.py
from __future__ import annotations
import numpy as np
from typing import Literal

TargetKind = Literal["R", "T", "phase_t", "phase_r"]

def _phase(z: np.ndarray | complex) -> np.ndarray:
    return np.angle(z)

def rms_merit(
    constants: dict,
    r: np.ndarray = np.array([]),
    t: np.ndarray = np.array([]),
    R: np.ndarray = np.array([]),
    T: np.ndarray = np.array([])
) -> float:
    """
    Универсальная RMS-мерит функция для многокритериальных целей.
    """
    errs = []

    if "R" in constants["targets"]:
        resid = (R - constants["targets"]["R"]["target"]) / constants["targets"]["R"]["sigma"]
        errs.append(resid**2)
    if "T" in constants["targets"]:
        resid = (T - constants["targets"]["T"]["target"]) / constants["targets"]["T"]["sigma"]
        errs.append(resid**2)

    if "phase_t" in constants["targets"] or "phase_r" in constants["targets"]:
        if constants["pol"] == "u":
            raise ValueError("Фазовые цели нельзя задавать при pol='u'; выберите 's' или 'p'.")
        if "phase_t" in constants["targets"]:
            resid = (_phase(t) - constants["targets"]["phase_t"]["target"]) / constants["targets"]["phase_t"]["sigma"]
            errs.append(resid**2)
        if "phase_r" in constants["targets"]:
            resid = (_phase(r) - constants["targets"]["phase_r"]["target"]) / constants["targets"]["phase_r"]["sigma"]
            errs.append(resid**2)

    if not errs:
        return 0.0
    resid_all = np.concatenate(errs, axis=0)
    return float(np.sqrt(np.mean(resid_all)))

def rms_merit_layers(
    constants: dict,
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

    if "R" in constants["targets"]:
        resid = (R - constants["targets"]["R"]["target"][None, :]) / constants["targets"]["R"]["sigma"][None, :]
        errs_all.append(resid**2)
    if "T" in constants["targets"]:
        resid = (T - constants["targets"]["T"]["target"][None, :]) / constants["targets"]["T"]["sigma"][None, :]
        errs_all.append(resid**2)

    if "phase_t" in constants["targets"] or "phase_r" in constants["targets"]:
        if constants["pol"] == "u":
            raise ValueError("Фазовые цели нельзя задавать при pol='u'.")
        if "phase_t" in constants["targets"]:
            resid = (np.angle(t) - constants["targets"]["phase_t"]["target"][None, :]) / constants["targets"]["phase_t"]["sigma"][None, :]
            errs_all.append(resid**2)
        if "phase_r" in constants["targets"]:
            resid = (np.angle(r) - constants["targets"]["phase_r"]["target"][None, :]) / constants["targets"]["phase_r"]["sigma"][None, :]
            errs_all.append(resid**2)

    if not errs_all:
        return np.zeros(r.shape[0])


    resid_all = np.concatenate(errs_all, axis=1)  # (num_layers, N_total)
    return np.sqrt(np.mean(resid_all, axis=1, dtype=np.float64))    # (num_layers,)
