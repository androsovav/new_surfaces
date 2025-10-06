# src/core/merit.py
from __future__ import annotations
import numpy as np

EPS_RATIO = 1e-12

def _phase(z: np.ndarray | complex) -> np.ndarray:
    return np.angle(z)

def _pol_list_from_targets(constants: dict, available_pols: list[str]) -> list[str]:
    if isinstance(constants.get("targets"), dict):
        keys = [k for k in ("s", "p") if k in constants["targets"]]
        if keys:
            return keys
    return [p for p in ("s", "p") if p in available_pols]

def rms_merit(
    constants: dict,
    r: dict[str, np.ndarray],
    t: dict[str, np.ndarray],
    R: dict[str, np.ndarray],
    T: dict[str, np.ndarray],
) -> float:
    errs = []
    pols = _pol_list_from_targets(constants, available_pols=list(R.keys()))
    tg = constants.get("targets", {})

    # Поляризационно-разделённые цели
    for pol in pols:
        tp = tg.get(pol, {})
        if "R" in tp:
            resid = (R[pol] - tp["R"]["target"]) / tp["R"]["sigma"]
            errs.append(resid**2)
        if "T" in tp:
            resid = (T[pol] - tp["T"]["target"]) / tp["T"]["sigma"]
            errs.append(resid**2)
        if "phase_t" in tp:
            resid = (_phase(t[pol]) - tp["phase_t"]["target"]) / tp["phase_t"]["sigma"]
            errs.append(resid**2)
        if "phase_r" in tp:
            resid = (_phase(r[pol]) - tp["phase_r"]["target"]) / tp["phase_r"]["sigma"]
            errs.append(resid**2)

    # Межполяризационные цели
    if "ratio_RsRp" in tg:
        if not ("s" in R and "p" in R):
            raise ValueError("ratio_RsRp требует наличия и R['s'], и R['p'].")
        ratio = (R["s"] + EPS_RATIO) / (R["p"] + EPS_RATIO)
        resid = (ratio - tg["ratio_RsRp"]["target"]) / tg["ratio_RsRp"]["sigma"]
        errs.append(resid**2)

    if not errs:
        return 0.0

    resid_all = np.concatenate([e.reshape(1, -1) for e in errs], axis=1)
    return float(np.sqrt(np.mean(resid_all)))

def rms_merit_layers(
    constants: dict,
    r: dict[str, np.ndarray],
    t: dict[str, np.ndarray],
    R: dict[str, np.ndarray],
    T: dict[str, np.ndarray],
) -> np.ndarray:
    """
    r[pol], t[pol], R[pol], T[pol] формы (num_layers, n_wavelengths).
    Возвращает (num_layers,).
    """
    errs_all = []
    pols = _pol_list_from_targets(constants, available_pols=list(R.keys()))
    tg = constants.get("targets", {})

    # Поляризационно-разделённые цели
    for pol in pols:
        tp = tg.get(pol, {})
        if "R" in tp:
            resid = (R[pol] - tp["R"]["target"][None, :]) / tp["R"]["sigma"][None, :]
            errs_all.append(resid**2)
        if "T" in tp:
            resid = (T[pol] - tp["T"]["target"][None, :]) / tp["T"]["sigma"][None, :]
            errs_all.append(resid**2)
        if "phase_t" in tp:
            resid = (np.angle(t[pol]) - tp["phase_t"]["target"][None, :]) / tp["phase_t"]["sigma"][None, :]
            errs_all.append(resid**2)
        if "phase_r" in tp:
            resid = (np.angle(r[pol]) - tp["phase_r"]["target"][None, :]) / tp["phase_r"]["sigma"][None, :]
            errs_all.append(resid**2)

    # Межполяризационные цели
    if "ratio_RsRp" in tg:
        if not ("s" in R and "p" in R):
            raise ValueError("ratio_RsRp требует наличия и R['s'], и R['p'].")
        ratio = (R["s"] + EPS_RATIO) / (R["p"] + EPS_RATIO)  # (L, K)
        resid = (ratio - tg["ratio_RsRp"]["target"][None, :]) / tg["ratio_RsRp"]["sigma"][None, :]
        errs_all.append(resid**2)

    if not errs_all:
        # Нет целей — нули по слоям
        any_pol = next(iter(R.keys()))
        return np.zeros(R[any_pol].shape[0], dtype=float)

    resid_all = np.concatenate(errs_all, axis=1)  # (L, N_sum)
    return np.sqrt(np.mean(resid_all, axis=1, dtype=np.float64))
