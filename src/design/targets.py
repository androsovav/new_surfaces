# src/design/targets.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, Dict, Literal

Pol = Literal["s", "p"]
Band = Tuple[float, float]

def _alloc_like(wl: np.ndarray, val: float) -> np.ndarray:
    return np.full_like(wl, float(val), dtype=float)

def _empty_targets(pols: Iterable[Pol]) -> Dict[Pol, dict]:
    return {pol: {} for pol in pols}

def combine_targets(*targets: dict) -> dict:
    """
    Сливает цели:
      - поляризационные блоки: {"s": {...}, "p": {...}}
      - глобальные блоки (межполяризационные): например {"ratio_RsRp": {...}}
    """
    out: dict = {}
    for tg in targets:
        for k, v in tg.items():
            if k in ("s", "p"):
                out.setdefault(k, {})
                out[k].update(v)
            else:
                # глобальные ключи (межполяризационные цели)
                out.setdefault(k, {})
                out[k].update(v)
    return out

def target_AR(wavelengths, R_target=0.0, sigma=0.01, pols=("s","p")) -> dict:
    tgt = _alloc_like(wavelengths, R_target)
    sig = _alloc_like(wavelengths, sigma)
    out = _empty_targets(pols)
    for pol in pols:
        out[pol]["R"] = {"target": tgt, "sigma": sig}
    return out

def target_bandpass(wavelengths, passbands: Iterable[Band],
                    sigma_pass=0.01, sigma_stop=0.01,
                    pols=("s","p"), T_in_pass=1.0, T_out=0.0) -> dict:
    target = _alloc_like(wavelengths, T_out)
    sigma  = _alloc_like(wavelengths, sigma_stop)
    for a, b in passbands:
        m = (wavelengths >= a) & (wavelengths <= b)
        target[m] = T_in_pass
        sigma[m]  = sigma_pass
    out = _empty_targets(pols)
    for pol in pols:
        out[pol]["T"] = {"target": target, "sigma": sigma}
    return out

def target_low_reflect(wavelengths, bands: Iterable[Band],
                       R_val=0.01, sigma=0.01, pols=("s","p"),
                       R_else=1.0, sigma_else=1.0) -> dict:
    target = _alloc_like(wavelengths, R_else)
    sig    = _alloc_like(wavelengths, sigma_else)
    m = np.zeros_like(wavelengths, dtype=bool)
    for a, b in bands:
        m |= (wavelengths >= a) & (wavelengths <= b)
    target[m] = R_val
    sig[m]    = sigma
    out = _empty_targets(pols)
    for pol in pols:
        out[pol]["R"] = {"target": target, "sigma": sig}
    return out

def target_T_bandpoint(wavelengths, wl0: float, halfwidth: float,
                       T_center=0.001, sigma_center=0.001, sigma_band=0.01,
                       pols=("s","p")) -> dict:
    target = np.zeros_like(wavelengths, dtype=float)
    sigma  = _alloc_like(wavelengths, sigma_band)
    idx0 = int(np.argmin(np.abs(wavelengths - wl0)))
    target[idx0] = T_center
    sigma[idx0]  = sigma_center
    out = _empty_targets(pols)
    for pol in pols:
        out[pol]["T"] = {"target": target, "sigma": sigma}
    return out

def target_ratio_RsRp(wavelengths,
                      bands: Iterable[Band],
                      ratio_target: float = 0.0,
                      sigma_in: float = 0.05,
                      sigma_out: float = np.inf) -> dict:
    """
    Межполяризационная цель: минимизировать Rs/Rp в заданных полосах.
    Возвращает глобальный блок:
      {"ratio_RsRp": {"target": ..., "sigma": ...}}
    Вне полос можно задать большую sigma (по умолчанию inf → без штрафа).
    """
    target = _alloc_like(wavelengths, ratio_target)
    sigma  = _alloc_like(wavelengths, sigma_out)
    m = np.zeros_like(wavelengths, dtype=bool)
    for a, b in bands:
        m |= (wavelengths >= a) & (wavelengths <= b)
    sigma[m] = sigma_in
    return {"ratio_RsRp": {"target": target, "sigma": sigma}}
