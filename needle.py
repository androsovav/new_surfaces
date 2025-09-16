# needle.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal, Dict, Any

from optics import (
    Stack, _M_layer, _cos_theta_in_layer, _q_parameter, _n_of,
    rt_amplitudes, _M_stack
)
from design import insert_layer, insert_with_split
from merit import rms_merit
from optimizers import coordinate_descent_thicknesses
from metrics import layer_count, total_optical_thickness

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
    где:
      left_iface[k]  = M0 * M1 * ... * M_{k-1}
      right_iface[k] = M_k * ... * M_{N-1}
      split i: слой i разбит пополам ⇒
        left_split[i]  = left_iface[i]  @ M_half(i)
        right_split[i] = M_half(i)      @ right_iface[i+1]
    """
    n_in = _n_of(stack.n_inc, wl)
    N = len(stack.layers)

    # префиксы интерфейсов
    left_iface = [np.eye(2, dtype=complex)]
    for j in range(N):
        Lj = stack.layers[j]
        nj = _n_of(Lj.n, wl)
        cosj = _cos_theta_in_layer(nj, n_in, theta_inc)
        left_iface.append(left_iface[-1] @ _M_layer(nj, Lj.d, wl, cosj, pol))

    # суффиксы интерфейсов
    right_iface = [None] * (N + 1)
    right_iface[N] = np.eye(2, dtype=complex)
    for j in reversed(range(N)):
        Lj = stack.layers[j]
        nj = _n_of(Lj.n, wl)
        cosj = _cos_theta_in_layer(nj, n_in, theta_inc)
        right_iface[j] = _M_layer(nj, Lj.d, wl, cosj, pol) @ right_iface[j + 1]

    # сплиты
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
    """
    Производная матрицы слоя по толщине в точке d=0:
      dM/dd|0 = [[0, i*k/q], [i*q*k, 0]]
    где k = 2π n cosθ / λ, q = n cosθ (s) или cosθ/n (p).
    """
    cos_new = _cos_theta_in_layer(n_new, n_in, theta_inc)
    q = _q_parameter(n_new, cos_new, pol)
    k = 2.0 * np.pi * n_new * cos_new / wl
    return np.array([[0.0+0.0j, 1j * k / q],
                     [1j * q * k, 0.0+0.0j]], dtype=complex)

def _dr_dt_from_dM(M: np.ndarray, q_in: complex, q_sub: complex, dM: np.ndarray) -> Tuple[complex, complex]:
    """
    Дифференциалы амплитуд r,t через дифференциал матрицы dM.
      r = A/B, t = 2*q_in / B,
      A = (m11 + m12*q_sub)*q_in - (m21 + m22*q_sub)
      B = (m11 + m12*q_sub)*q_in + (m21 + m22*q_sub)
    """
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
# P-КАРТЫ
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
) -> Tuple[List[Tuple[PositionKind,int]], np.ndarray, np.ndarray]:
    """
    Аналитическая P-карта: d(MF)/dd в точке d=0 для вставляемого слоя.
    Поддержка целей: R, T, phase_t, phase_r.
    Для pol='u' (неполяриз.) — усреднение по s/p для R,T (фазовые цели при 'u' не поддерживаются).
    """
    if pol == "u" and (("phase_t" in targets) or ("phase_r" in targets)):
        raise ValueError("Фазовые цели недоступны при pol='u'. Используйте 's' или 'p'.")

    # Базовый MF и общее число резидуалов L (для нормировки производной MF)
    base_MF = rms_merit(stack, wavelengths, targets, pol=pol, theta_inc=theta_inc)
    if base_MF < 1e-18:
        positions = enumerate_positions(stack)
        return positions, np.full(len(positions), np.nan), np.zeros(len(positions))

    L_total = 0
    if "R" in targets:       L_total += len(wavelengths)
    if "T" in targets:       L_total += len(wavelengths)
    if "phase_t" in targets: L_total += len(wavelengths)
    if "phase_r" in targets: L_total += len(wavelengths)

    positions = enumerate_positions(stack)
    n_best = np.full(len(positions), np.nan, dtype=float)
    dmf_best = np.full(len(positions), np.inf, dtype=float)

    # Поляризации: либо одна (s/p), либо обе (u → усреднение по s/p)
    pols = ["s", "p"] if pol == "u" else [pol]

    # КЭШ: базовые величины для каждой λ и поляризации
    base_cache: Dict[Tuple[str,int], Dict[str, Any]] = {}
    for p in pols:
        for il, wl in enumerate(wavelengths):
            wl = float(wl)
            M_full, q_in, q_sub = _M_stack(stack, wl, theta_inc, p)
            r0, t0 = rt_amplitudes(stack, wl, theta_inc, p)
            left_iface, right_iface, left_split, right_split = _prefix_suffix_mats_for_interfaces_and_splits(
                stack, wl, p, theta_inc
            )
            base_cache[(p, il)] = dict(
                wl=wl, M_full=M_full, q_in=q_in, q_sub=q_sub, r0=r0, t0=t0,
                left_iface=left_iface, right_iface=right_iface,
                left_split=left_split, right_split=right_split
            )

    has_R  = "R" in targets
    has_T  = "T" in targets
    has_pt = "phase_t" in targets
    has_pr = "phase_r" in targets

    # Перебор позиций и кандидатов материала (с переиспользованием кэша)
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

                    # левая/правая часть матрицы в точке вставки
                    if pos[0] == "interface":
                        left  = c["left_iface"][pos[1]]
                        right = c["right_iface"][pos[1]]
                    else:
                        left  = c["left_split"][pos[1]]
                        right = c["right_split"][pos[1]]

                    # дифференциал матрицы вставляемого слоя
                    n_in = _n_of(stack.n_inc, wl)
                    dM_layer  = _dM_layer_dd_at_zero(n_new, n_in, wl, p, theta_inc)
                    dM_total  = left @ dM_layer @ right

                    # производные амплитуд
                    dr, dt = _dr_dt_from_dM(M_full, q_in, q_sub, dM_total)

                    # dR и dT
                    dR = 2.0 * np.real(np.conj(r0) * dr)
                    dT = np.real(q_sub / q_in) * 2.0 * np.real(np.conj(t0) * dt)

                    # вклад в Σ resid * dresid/dd
                    if has_R:
                        R0 = np.abs(r0)**2
                        resid  = (R0 - targets["R"]["target"][il]) / targets["R"]["sigma"][il]
                        dresid = dR / targets["R"]["sigma"][il]
                        accum += resid * dresid
                    if has_T:
                        T0 = np.real(q_sub / q_in) * np.abs(t0)**2
                        resid  = (T0 - targets["T"]["target"][il]) / targets["T"]["sigma"][il]
                        dresid = dT / targets["T"]["sigma"][il]
                        accum += resid * dresid
                    if has_pt:
                        # d arg(t)/dd = Im(conj(t0)*dt)/|t0|^2
                        denom  = max(np.abs(t0)**2, 1e-30)
                        dphi_t = np.imag(np.conj(t0) * dt) / denom
                        resid  = (_phase(t0) - targets["phase_t"]["target"][il]) / targets["phase_t"]["sigma"][il]
                        dresid = dphi_t / targets["phase_t"]["sigma"][il]
                        accum += resid * dresid
                    if has_pr:
                        # d arg(r)/dd = Im(conj(r0)*dr)/|r0|^2
                        denom  = max(np.abs(r0)**2, 1e-30)
                        dphi_r = np.imag(np.conj(r0) * dr) / denom
                        resid  = (_phase(r0) - targets["phase_r"]["target"][il]) / targets["phase_r"]["sigma"][il]
                        dresid = dphi_r / targets["phase_r"]["sigma"][il]
                        accum += resid * dresid

            if pol == "u":
                accum *= 0.5  # усреднение s/p

            dMF_dd = (1.0 / (L_total * base_MF)) * accum

            if dMF_dd < best_val:
                best_val = dMF_dd
                best_n   = float(np.real(n_new))

        n_best[idx_pos] = best_n
        dmf_best[idx_pos] = best_val

    return positions, n_best, dmf_best


# ---------------------------
# ОСНОВНОЙ ЦИКЛ
# ---------------------------

def needle_cycle(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: List[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    # выбор P-карты
    pmap: PMapKind = "analytic",      # "analytic" | "discrete"
    # параметры "иглы"
    d_init: float = 2e-9,
    d_eps: float = 5e-10,
    # локальная доводка
    coord_step_rel: float = 0.25,
    coord_min_step_rel: float = 0.01,
    coord_iters: int = 60,
    d_min: float | None = 0.5e-9,
    d_max: float | None = None,
    # остановы
    max_steps: int = 30,
    min_rel_improv: float = 1e-3,
    max_layers: int | None = None,
    max_tot_nmopt: float | None = None,
    wl_ref_for_tot: float = 550e-9,
    # вывод
    verbose: bool = True,
) -> Tuple[Stack, Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    mf = rms_merit(stack, wavelengths, targets, pol=pol, theta_inc=theta_inc)
    N = layer_count(stack)
    TOT_nmopt = total_optical_thickness(stack, wl_ref_for_tot) / 1e-9
    history.append({"step": 0, "action": "init", "MF": mf, "N": N, "TOT_nmopt": TOT_nmopt})
    if verbose:
        print(f"step 00: init | MF={mf:.4f}, N={N}, TOT={TOT_nmopt:.1f}")

    for step in range(1, max_steps + 1):
        if pmap == "analytic":
            positions, n_best, dmf_best = analytic_excitation_map(
                stack, wavelengths, targets, n_candidates,
                pol=pol, theta_inc=theta_inc, d_min=d_min
            )
        else:
            positions, n_best, dmf_best = discrete_excitation_map(
                stack, wavelengths, targets, n_candidates,
                pol=pol, theta_inc=theta_inc, d_eps=d_eps, d_min=d_min
            )

        j = int(np.argmin(dmf_best))
        if not np.isfinite(dmf_best[j]) or dmf_best[j] >= 0.0:
            history.append({"step": step, "action": "stop_no_negative_P", "best_dMF": float(dmf_best[j])})
            if verbose:
                print(f"step {step:02d}: stop_no_negative_P | best_dMF={dmf_best[j]:.3e}")
            break

        pos = positions[j]
        d_insert = max(d_init, (d_min if d_min is not None else d_init))
        stack = _test_insert(stack, pos, n_new=float(n_best[j]), d_new=d_insert)

        stack, mf_new = coordinate_descent_thicknesses(
            stack, wavelengths, targets, pol=pol, theta_inc=theta_inc,
            step_rel=coord_step_rel, min_step_rel=coord_min_step_rel, iters=coord_iters,
            d_min=d_min, d_max=d_max
        )

        N = layer_count(stack)
        TOT_nmopt = total_optical_thickness(stack, wl_ref_for_tot) / 1e-9
        rel_improv = (mf - mf_new) / max(mf, 1e-12)
        history.append({
            "step": step, "action": "needle+optimize",
            "pos": pos, "n_new": float(n_best[j]), "d_mf_pred": float(dmf_best[j]),
            "MF": float(mf_new), "MF_prev": float(mf), "rel_improv": float(rel_improv),
            "N": int(N), "TOT_nmopt": float(TOT_nmopt),
        })
        if verbose:
            kind, idx = pos
            print(
                f"step {step:02d}: add@{kind}[{idx}] n={n_best[j]:.2f} "
                f"{'d(MF)/dd≈' if pmap=='analytic' else 'dMF≈'}{dmf_best[j]:.3e} -> "
                f"MF {mf:.4f} → {mf_new:.4f} (Δrel={rel_improv*100:.2f}%), "
                f"N={N}, TOT={TOT_nmopt:.1f}"
            )

        mf = mf_new

        if rel_improv < min_rel_improv:
            history.append({"step": step, "action": "stop_small_improv", "rel_improv": float(rel_improv)})
            if verbose:
                print(f"step {step:02d}: stop_small_improv | rel_improv={rel_improv:.3e}")
            break
        if max_layers is not None and N > max_layers:
            history.append({"step": step, "action": "stop_layers_limit", "N": int(N), "max_layers": int(max_layers)})
            if verbose:
                print(f"step {step:02d}: stop_layers_limit | N={N} > {max_layers}")
            break
        if max_tot_nmopt is not None and TOT_nmopt > max_tot_nmopt:
            history.append({"step": step, "action": "stop_tot_limit", "TOT_nmopt": float(TOT_nmopt), "max_tot_nmopt": float(max_tot_nmopt)})
            if verbose:
                print(f"step {step:02d}: stop_tot_limit | TOT={TOT_nmopt:.1f} > {max_tot_nmopt}")
            break

    return stack, {"history": history}
