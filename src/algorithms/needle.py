# src/algorithms/needle.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal, Dict, Any

from ..core.optics import (
    Stack, make_M, cos_theta_in_layer, q_parameter, n_of,
    rt_amplitudes, RT_coeffs
)
from ..design.design import insert_layer, insert_with_split
from ..core.merit import rms_merit
from .optimizers import coordinate_descent_thicknesses
from ..core.metrics import layer_count, total_optical_thickness

PositionKind = Literal["interface", "split"]
PMapKind = Literal["discrete", "analytic"]

# ---------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------

def _test_insert(stack: Stack, pos: Tuple[PositionKind, int], n_new: float, d_new: float) -> Stack:
    kind, idx = pos
    if kind == "interface":
        return insert_layer(stack, index=idx, n_new=n_new, d_new=d_new)
    else:
        return insert_with_split(stack, layer_index=idx, n_new=n_new, d_new=d_new, split_ratio=0.5)

def enumerate_positions(stack: Stack) -> List[Tuple[PositionKind, int]]:
    N = len(stack.layers)
    positions: List[Tuple[PositionKind, int]] = []
    positions += [("interface", i) for i in range(N + 1)]
    positions += [("split", i) for i in range(N)]
    return positions

def _dM_layer_dd_at_zero(n_new: complex, n_in: complex, wl: float, pol: str, theta_inc: float) -> np.ndarray:
    cos_new = cos_theta_in_layer(n_new, n_in, theta_inc)
    q = q_parameter(n_new, cos_new, pol)
    k = 2.0 * np.pi * n_new * cos_new / wl
    return np.array([[0.0+0.0j, 1j * k / q],
                     [1j * q * k, 0.0+0.0j]], dtype=complex)

def _dr_dt_from_dM(M: np.ndarray, q_in: complex, q_sub: complex, dM: np.ndarray) -> Tuple[complex, complex]:
    m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
    dm11, dm12, dm21, dm22 = dM[0,0], dM[0,1], dM[1,0], dM[1,1]
    A = (m11 + m12*q_sub)*q_in - (m21 + m22*q_sub)
    B = (m11 + m12*q_sub)*q_in + (m21 + m22*q_sub)
    dA = (dm11 + dm12*q_sub)*q_in - (dm21 + dm22*q_sub)
    dB = (dm11 + dm12*q_sub)*q_in + (dm21 + dm22*q_sub)
    r = A / B
    dr = (dA * B - A * dB) / (B * B)
    dt = - (2.0 * q_in * dB) / (B * B)
    return dr, dt

def _phase(z: np.ndarray | complex) -> np.ndarray:
    return np.angle(z)

# ---------------------------
# P-КАРТА
# ---------------------------

def analytic_excitation_map(
    stack: Stack,
    q_in: np.ndarray, q_sub: np.ndarray,
    qH: np.ndarray, qL: np.ndarray,
    nH_values: np.ndarray, nL_values: np.ndarray,
    wavelengths: np.ndarray,
    targets: dict,
    pol: Literal["s", "p"],
    theta_inc: np.ndarray,
    d_min: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованная аналитическая P-карта:
    - середины слоёв
    - перед первым слоем
    - после последнего слоя
    Для H вставляем L, для L вставляем H.
    """

    base_MF = rms_merit(stack, q_in, q_sub, wavelengths, targets, pol, theta_inc)
    if base_MF < 1e-18:
        return np.arange(-1, len(stack.layers)+1), np.full(len(stack.layers)+2, np.nan), np.zeros(len(stack.layers)+2)

    N = len(stack.layers)
    positions = list(range(N)) + [-1, N]   # середины + границы
    n_best = np.empty(len(positions), dtype=object)
    dmf_best = np.empty(len(positions), dtype=float)

    for k, pos in enumerate(positions):
        if pos == -1:  # перед первым
            ins_litera = "L" if stack.layers[0].litera == "H" else "H"
        elif pos == N:  # после последнего
            ins_litera = "L" if stack.layers[-1].litera == "H" else "H"
        else:  # середина слоя
            ins_litera = "L" if stack.layers[pos].litera == "H" else "H"

        # подбираем набор n и q
        if ins_litera == "H":
            n_ins, q_ins = nH_values, qH
        else:
            n_ins, q_ins = nL_values, qL

        # матрица вставки
        phi_ins = 2.0 * np.pi * n_ins * d_min / wavelengths
        sphi, cphi = np.sin(phi_ins), np.cos(phi_ins)
        M_ins = make_M(sphi, cphi, q_ins, len(wavelengths))  # (2,2,K)

        # собранная матрица
        if pos == -1:  # перед первым
            pref = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,len(wavelengths)))
            suff = stack.M
            pref, mins, suff = pref.transpose(2,0,1), M_ins.transpose(2,0,1), suff.transpose(2,0,1)
        elif pos == N:  # после последнего
            pref = stack.M
            suff = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,len(wavelengths)))
            pref, mins, suff = pref.transpose(2,0,1), M_ins.transpose(2,0,1), suff.transpose(2,0,1)
        else:  # середина
            pref = stack.prefix[pos].transpose(2,0,1)
            mins = M_ins.transpose(2,0,1)
            suff = stack.suffix[pos].transpose(2,0,1)

        M_tot = np.einsum("lij,ljk,lkm->lim", pref, mins, suff).transpose(1,2,0)

        # амплитуды и коэффициенты
        r, t = rt_amplitudes(M_tot, q_in, q_sub)
        R, T = RT_coeffs(r, t, q_in, q_sub)

        tmp_stack = Stack(layers=stack.layers, prefix=stack.prefix, suffix=stack.suffix,
                          M=M_tot, r=r, t=t, R=R, T=T)
        mf_new = rms_merit(tmp_stack, q_in, q_sub, wavelengths, targets, pol, theta_inc)

        dmf_best[k] = base_MF - mf_new
        n_best[k] = ins_litera

    return np.array(positions), n_best, dmf_best




# ---------------------------
# NEEDLE-CYCLE
# ---------------------------

# src/algorithms/needle.py (фрагмент)

def needle_cycle(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    nH_values: np.ndarray,
    nL_values: np.ndarray,
    q_in: np.ndarray,
    q_sub: np.ndarray,
    qH: np.ndarray,
    qL: np.ndarray,
    pol: str,
    theta_inc: np.ndarray,
    d_init: float = 2e-9,
    d_eps: float = 5e-10,
    coord_step_rel: float = 0.25,
    coord_min_step_rel: float = 0.02,
    coord_iters: int = 10,
    d_min: float | None = None,
    d_max: float | None = None,
    max_steps: int = 20,
    min_rel_improv: float = 1e-3,
    max_layers: int = 200,
    max_tot_nmopt: float = 1e9,
    wl_ref_for_tot: float = 550e-9,
    verbose: bool = False,
    log_timing: bool = False,
) -> Tuple[Stack, Dict[str, Any]]:
    """
    Needle-оптимизация с кэшированием промежуточных расчетов для ускорения.
    """
    import time
    t0 = time.perf_counter()

    current = stack
    history: List[Dict[str, Any]] = []

    # основной цикл по шагам
    for step in range(max_steps):
        mf_before = rms_merit(current, q_in, q_sub, wavelengths, targets, pol=pol, theta_inc=theta_inc)
        print("mf_before")
        print(mf_before)
        # ======================
        # 1. построение P-карты
        # ======================
        positions, n_best, dmf_best = analytic_excitation_map(
            current, q_in, q_sub, qH, qL, nH_values, nL_values,
            wavelengths, targets,
            pol, theta_inc, d_min
        )
        print(positions)
        print(n_best)
        print(dmf_best)

        # выбор лучшей позиции
        idx_best = int(np.argmin(dmf_best))
        if not np.isfinite(dmf_best[idx_best]):
            break
        pos_best = positions[idx_best]
        n_new = n_best[idx_best]
        print(pos_best)
        print(n_new)

        # ============================
        # 2. вставка нового слоя
        # ============================
        current = _test_insert(current, pos_best, n_new=n_new, d_new=d_init)

        # ============================
        # 3. локальная оптимизация
        # ============================
        current, mf_after = coordinate_descent_thicknesses(
            current, wavelengths, targets,
            pol=pol, theta_inc=theta_inc,
            step_rel=coord_step_rel,
            min_step_rel=coord_min_step_rel,
            iters=coord_iters,
            d_min=d_min, d_max=d_max,
        )

        history.append({"step": step, "MF": mf_after})

        # критерии останова
        if mf_before - mf_after < min_rel_improv * mf_before:
            break
        if layer_count(current) > max_layers:
            break
        if total_optical_thickness(current, wl_ref_for_tot) > max_tot_nmopt:
            break

    elapsed = time.perf_counter() - t0 if log_timing else None
    return current, {"history": history, "elapsed": elapsed}

