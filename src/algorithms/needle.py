# src/algorithms/needle.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal, Dict, Any

from ..core.optics import (
    Stack, _M_layer, _cos_theta_in_layer, _q_parameter, _n_of,
    rt_amplitudes, _M_stack
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

def _prefix_suffix_mats_for_interfaces_and_splits(
    stack: Stack, wl: float, pol: str, theta_inc: float
):
    """
    Возвращает:
      left_iface[k], right_iface[k]  для k=0..N
      left_split[i], right_split[i]  для i=0..N-1
    """
    n_in = _n_of(stack.n_inc, wl)
    N = len(stack.layers)

    left_iface = [np.eye(2, dtype=complex)]
    for j in range(N):
        Lj = stack.layers[j]
        nj = _n_of(Lj.n, wl)
        cosj = _cos_theta_in_layer(nj, n_in, theta_inc)
        left_iface.append(left_iface[-1] @ _M_layer(nj, Lj.d, wl, cosj, pol))

    right_iface = [None] * (N + 1)
    right_iface[N] = np.eye(2, dtype=complex)
    for j in reversed(range(N)):
        Lj = stack.layers[j]
        nj = _n_of(Lj.n, wl)
        cosj = _cos_theta_in_layer(nj, n_in, theta_inc)
        right_iface[j] = _M_layer(nj, Lj.d, wl, cosj, pol) @ right_iface[j + 1]

    left_split = []
    right_split = []
    for i in range(N):
        Li = stack.layers[i]
        ni = _n_of(Li.n, wl)
        cosi = _cos_theta_in_layer(ni, n_in, theta_inc)
        M_half = _M_layer(ni, Li.d * 0.5, wl, cosi, pol)
        left_split.append(left_iface[i] @ M_half)
        right_split.append(M_half @ right_iface[i + 1])

    return left_iface, right_iface, left_split, right_split

def _dM_layer_dd_at_zero(n_new: complex, n_in: complex, wl: float, pol: str, theta_inc: float) -> np.ndarray:
    cos_new = _cos_theta_in_layer(n_new, n_in, theta_inc)
    q = _q_parameter(n_new, cos_new, pol)
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
# P-КАРТЫ (discrete и analytic)
# ---------------------------

def discrete_excitation_map(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: List[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    d_eps: float = 5e-10,
    d_min: float | None = None,
) -> Tuple[List[Tuple[PositionKind,int]], np.ndarray, np.ndarray]:
    """
    Дискретная P-карта: ΔMF при тестовой вставке сверхтонкого слоя.
    """
    base_mf = rms_merit(stack, wavelengths, targets, pol=pol, theta_inc=theta_inc)
    positions = enumerate_positions(stack)
    n_best = np.full(len(positions), np.nan, dtype=float)
    dmf_best = np.full(len(positions), np.inf, dtype=float)

    for i, pos in enumerate(positions):
        for n_new in n_candidates:
            if d_min is not None and d_eps < d_min:
                continue
            test = _test_insert(stack, pos, n_new=n_new, d_new=d_eps)
            mf = rms_merit(test, wavelengths, targets, pol=pol, theta_inc=theta_inc)
            dmf = mf - base_mf
            if dmf < dmf_best[i]:
                dmf_best[i] = dmf
                n_best[i] = n_new
    return positions, n_best, dmf_best

def analytic_excitation_map(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: List[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    d_min: float | None = None,
    base_cache: Dict[Tuple[str,int], Dict[str,Any]] | None = None,  # <─ новое
) -> Tuple[List[Tuple[PositionKind,int]], np.ndarray, np.ndarray]:
    """
    Аналитическая P-карта: d(MF)/dd в точке d=0 для вставляемого слоя.
    Поддержка целей: R, T, phase_t, phase_r.
    """
    if pol == "u" and (("phase_t" in targets) or ("phase_r" in targets)):
        raise ValueError("Фазовые цели недоступны при pol='u'. Используйте 's' или 'p'.")

    base_MF = rms_merit(stack, wavelengths, targets, pol=pol, theta_inc=theta_inc)
    if base_MF < 1e-18:
        positions = enumerate_positions(stack)
        return positions, np.full(len(positions), np.nan), np.zeros(len(positions))

    positions = enumerate_positions(stack)
    n_best = np.full(len(positions), np.nan, dtype=float)
    dmf_best = np.full(len(positions), np.inf, dtype=float)

    pols = ["s", "p"] if pol == "u" else [pol]

    # если кэш не передан ─ строим локально
    if base_cache is None:
        base_cache = {}
        for p in pols:
            for il, wl in enumerate(wavelengths):
                wl = float(wl)
                M_full, q_in, q_sub = _M_stack(stack, wl, theta_inc, p)
                r0, t0 = rt_amplitudes(stack, wl, theta_inc, p)
                left_iface, right_iface, left_split, right_split = _prefix_suffix_mats_for_interfaces_and_splits(
                    stack, wl, p, theta_inc
                )
                base_cache[(p, il)] = dict(
                    wl=wl, M_full=M_full, q_in=q_in, q_sub=q_sub,
                    r0=r0, t0=t0,
                    left_iface=left_iface, right_iface=right_iface,
                    left_split=left_split, right_split=right_split,
                )

    has_R  = "R" in targets
    has_T  = "T" in targets
    has_pt = "phase_t" in targets
    has_pr = "phase_r" in targets

    for idx_pos, pos in enumerate(positions):
        best_val = np.inf
        best_n   = np.nan

        for n_new_val in n_candidates:
            n_new = complex(n_new_val)
            accum = 0.0

            for p in pols:
                for il, _ in enumerate(wavelengths):
                    c = base_cache[(p, il)]
                    wl     = c["wl"]
                    M_full = c["M_full"]
                    q_in   = c["q_in"]
                    q_sub  = c["q_sub"]
                    r0     = c["r0"]
                    t0     = c["t0"]

                    if pos[0] == "interface":
                        left  = c["left_iface"][pos[1]]
                        right = c["right_iface"][pos[1]]
                    else:
                        left  = c["left_split"][pos[1]]
                        right = c["right_split"][pos[1]]

                    # матрица dM/dd при d=0
                    dM = _dM_layer_dd_at_zero(n_new, _n_of(stack.n_inc, wl), wl, p, theta_inc)
                    M_new = left @ dM @ right

                    dr, dt = _dr_dt_from_dM(M_full, q_in, q_sub, M_new)

                    if has_R:
                        accum += np.real(2.0 * np.conj(r0) * dr)
                    if has_T:
                        accum += np.real(2.0 * np.conj(t0) * dt)
                    if has_pt:
                        accum += 2.0 * (np.imag(np.conj(t0) * dt))
                    if has_pr:
                        accum += 2.0 * (np.imag(np.conj(r0) * dr))

            if accum < best_val:
                best_val = accum
                best_n   = n_new_val

        dmf_best[idx_pos] = best_val
        n_best[idx_pos]   = best_n

    return positions, n_best, dmf_best


# ---------------------------
# NEEDLE-CYCLE
# ---------------------------

# src/algorithms/needle.py (фрагмент)

def needle_cycle(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: List[float],
    pol: str = "s",
    theta_inc: float = 0.0,
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
    pmap: PMapKind = "analytic",  # "analytic" | "discrete"
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
        mf_before = rms_merit(current, wavelengths, targets, pol=pol, theta_inc=theta_inc)

        # ======================
        # 1. построение P-карты
        # ======================
        if pmap == "discrete":
            positions, n_best, dmf_best = discrete_excitation_map(
                current, wavelengths, targets, n_candidates,
                pol=pol, theta_inc=theta_inc, d_eps=d_eps, d_min=d_min,
            )
        else:
            # кешируем базовые матрицы для всех λ
            base_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
            pols = ["s", "p"] if pol == "u" else [pol]
            for p in pols:
                for il, wl in enumerate(wavelengths):
                    wl = float(wl)
                    M_full, q_in, q_sub = _M_stack(current, wl, theta_inc, p)
                    r0, t0_amp = rt_amplitudes(current, wl, theta_inc, p)
                    left_iface, right_iface, left_split, right_split = _prefix_suffix_mats_for_interfaces_and_splits(
                        current, wl, p, theta_inc
                    )
                    base_cache[(p, il)] = dict(
                        wl=wl, M_full=M_full, q_in=q_in, q_sub=q_sub,
                        r0=r0, t0=t0_amp,
                        left_iface=left_iface, right_iface=right_iface,
                        left_split=left_split, right_split=right_split,
                    )

            positions, n_best, dmf_best = analytic_excitation_map(
                current, wavelengths, targets, n_candidates,
                pol=pol, theta_inc=theta_inc, d_min=d_min,
            )

        # выбор лучшей позиции
        idx_best = int(np.argmin(dmf_best))
        if not np.isfinite(dmf_best[idx_best]):
            break
        pos_best = positions[idx_best]
        n_new = n_best[idx_best]

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

