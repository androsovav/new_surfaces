# src/algorithms/needle.py
from __future__ import annotations
import numpy as np
import time
from typing import List, Tuple, Literal, Dict, Any

from ..core.optics import (
    Layer, Stack, phi_parameter, make_M, rt_amplitudes, RT_coeffs
)
from ..core.merit import rms_merit
from ..algorithms.optimizers import coordinate_descent_thicknesses
from ..core.metrics import layer_count, total_optical_thickness


# ---------------------------
# Вспомогательная сборка стека по произвольной H/L-последовательности
# (аналог design.make_stack, но принимает letters[])
# ---------------------------
def make_stack_from_letters(
    letters: List[Literal["H","L"]],
    thickness: np.ndarray,
    n_inc_values: np.ndarray,
    n_sub_values: np.ndarray,
    nH_values: np.ndarray,
    nL_values: np.ndarray,
    cos_theta_in_H_layers: np.ndarray,
    cos_theta_in_L_layers: np.ndarray,
    q_in: np.ndarray,
    q_sub: np.ndarray,
    qH: np.ndarray,
    qL: np.ndarray,
    wavelengths: np.ndarray,
    n_wavelengths: int,
    pol: Literal["s","p"],
) -> Stack:
    """
    Полностью повторяет логику сборки из design.make_stack, но вместо
    автоматического чередования использует заданный массив букв.
    """
    num_layers = len(thickness)
    layers = np.empty(num_layers, dtype=object)

    for i, litera in enumerate(letters):
        if litera == "H":
            phi = phi_parameter(nH_values, float(thickness[i]), cos_theta_in_H_layers, wavelengths)
            sphi, cphi = np.sin(phi), np.cos(phi)
            M = make_M(sphi, cphi, qH, n_wavelengths)
        else:
            phi = phi_parameter(nL_values, float(thickness[i]), cos_theta_in_L_layers, wavelengths)
            sphi, cphi = np.sin(phi), np.cos(phi)
            M = make_M(sphi, cphi, qL, n_wavelengths)
        layers[i] = Layer(litera=litera, d=float(thickness[i]), phi=phi, sphi=sphi, cphi=cphi, M=M)

    # Префиксы/суффиксы на «половинках» слоёв — как в design.make_stack
    prefix = np.empty((num_layers, 2, 2, n_wavelengths), dtype=complex)
    suffix = np.empty((num_layers, 2, 2, n_wavelengths), dtype=complex)
    left = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, n_wavelengths))
    right = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, n_wavelengths))

    for i in range(num_layers):
        L = layers[i]; half_d = 0.5 * L.d
        if L.litera == "H":
            phi_half = phi_parameter(nH_values, half_d, cos_theta_in_H_layers, wavelengths)
            M_half = make_M(np.sin(phi_half), np.cos(phi_half), qH, n_wavelengths)
        else:
            phi_half = phi_parameter(nL_values, half_d, cos_theta_in_L_layers, wavelengths)
            M_half = make_M(np.sin(phi_half), np.cos(phi_half), qL, n_wavelengths)

        prefix[i] = np.einsum('ijk,jlk->ilk', left, M_half)
        left = np.einsum('ijk,jlk->ilk', left, L.M)

        Lr = layers[-(i+1)]; half_dr = 0.5 * Lr.d
        if Lr.litera == "H":
            phi_half_r = phi_parameter(nH_values, half_dr, cos_theta_in_H_layers, wavelengths)
            M_half_r = make_M(np.sin(phi_half_r), np.cos(phi_half_r), qH, n_wavelengths)
        else:
            phi_half_r = phi_parameter(nL_values, half_dr, cos_theta_in_L_layers, wavelengths)
            M_half_r = make_M(np.sin(phi_half_r), np.cos(phi_half_r), qL, n_wavelengths)

        suffix[-(i+1)] = np.einsum('ijk,jlk->ilk', M_half_r, right)
        right = np.einsum('ijk,jlk->ilk', Lr.M, right)

    M_tot = left
    r, t = rt_amplitudes(M_tot, q_in, q_sub)
    R, T = RT_coeffs(r, t, q_in, q_sub)
    return Stack(layers=layers, prefix=prefix, suffix=suffix, M=M_tot, r=r, t=t, R=R, T=T)


# ---------------------------
# Производная матрицы слоя по толщине в нуле (игольчатая вариация)
# ---------------------------
def _dM_layer_dd_at_zero(q: np.ndarray, k: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    ∂M_layer/∂d |_{d=0} для всех λ: форма (2,2,K).
    Здесь k = 2π n cosθ / λ; при d→0: cosφ≈1, sinφ≈φ=k d → dM/dd:
      d/d(d) [ [cosφ, i sinφ / q], [i q sinφ, cosφ] ]_{d=0}
      = [ [0, i k / q], [i q k, 0] ].
    """
    K = len(wavelengths)
    dM = np.zeros((2, 2, K), dtype=complex)
    dM[0, 1, :] = 1j * k / q
    dM[1, 0, :] = 1j * q * k
    return dM


def _dr_dt_from_dM_vec(
    M: np.ndarray,        # shape: (K, 2, 2)
    q_in: np.ndarray,     # shape: (K,)
    q_sub: np.ndarray,    # shape: (K,)
    dM: np.ndarray        # shape: (K, 2, 2)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Векторизованная производная r,t по малой вариации M → M + ε dM.
    Ожидаем M и dM как (K,2,2). Возвращаем dr, dt формы (K,).
    Формулы: r = A/B, t = 2 q_in / B, где
      A = (m11 + m12 q_sub) q_in - (m21 + m22 q_sub)
      B = (m11 + m12 q_sub) q_in + (m21 + m22 q_sub)
    """
    # Разворачиваем элементы по K
    m11 = M[:, 0, 0]; m12 = M[:, 0, 1]
    m21 = M[:, 1, 0]; m22 = M[:, 1, 1]

    dm11 = dM[:, 0, 0]; dm12 = dM[:, 0, 1]
    dm21 = dM[:, 1, 0]; dm22 = dM[:, 1, 1]

    A  = (m11 + m12 * q_sub) * q_in - (m21 + m22 * q_sub)
    B  = (m11 + m12 * q_sub) * q_in + (m21 + m22 * q_sub)
    dA = (dm11 + dm12 * q_sub) * q_in - (dm21 + dm22 * q_sub)
    dB = (dm11 + dm12 * q_sub) * q_in + (dm21 + dm22 * q_sub)

    # r = A/B; t = 2 q_in / B
    # dr = (dA*B - A*dB) / B^2
    dr = (dA * B - A * dB) / (B * B)
    # dt = - (2 q_in dB) / B^2
    dt = -(2.0 * q_in * dB) / (B * B)
    return dr, dt


# ---------------------------
# Аналитическая P-карта (по Тихонравову)
# ---------------------------
def analytic_excitation_map(
    stack: Stack,
    q_in: np.ndarray, q_sub: np.ndarray,
    qH: np.ndarray, qL: np.ndarray,
    kH: np.ndarray, kL: np.ndarray,
    nH_values: np.ndarray, nL_values: np.ndarray,
    n_inc_values: np.ndarray,
    wavelengths: np.ndarray,
    targets: dict,
    pol: Literal["s","p"],
    theta_inc: float,
    d_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Для каждой потенциальной позиции (перед первым, середины всех слоёв, после последнего)
    и для «иглы» комплементарного материала (H↔L) оценивает линейный прирост ΔMF при добавлении
    тонкого слоя толщины e=d_eps. Возвращает:
      positions: [-1, 0, 1, ..., N]  (−1 перед первым, N после последнего, i — середина слоя i)
      letters:   ['H'|'L'] — какая «игла» даёт наилучший выигрыш
      dmf:       массив ожидаемых уменьшений MF (чем больше — тем лучше)
    Теория и физическая интерпретация метода «игл» — A.V. Tikhonravov et al. (1996/1997). :contentReference[oaicite:6]{index=6}
    """
    base_MF = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, stack.r, stack.t, stack.R, stack.T)
    N = len(stack.layers)
    positions = [-1] + list(range(N)) + [N]

    letters_best = np.empty(len(positions), dtype=object)
    dmf_best = np.full(len(positions), -np.inf, dtype=float)

    # Базовый M(λ) и удобные формы (K,2,2)
    M_tot = np.transpose(stack.M, (2,0,1))           # (K,2,2)
    I_K = np.tile(np.eye(2, dtype=complex)[None, :, :], (len(wavelengths), 1, 1))

    alpha = np.real(q_sub) / np.real(q_in)           # для T = alpha * |t|^2

    for k, pos in enumerate(positions):
        # Выбор комплементарной «иглы» (H↔L) в этой позиции
        if pos == -1:
            ins = "L" if stack.layers[0].litera == "H" else "H"
        elif pos == N:
            ins = "L" if stack.layers[-1].litera == "H" else "H"
        else:
            ins = "L" if stack.layers[pos].litera == "H" else "H"

        # Материальные параметры «иглы»
        if ins == "H":
            q_ins, k_ins = qH, kH
        else:
            q_ins, k_ins = qL, kL

        # ∂M_иглы/∂d |_{0} для всех λ: (2,2,K) → (K,2,2)
        dM_layer = np.transpose(_dM_layer_dd_at_zero(q_ins, k_ins, wavelengths), (2,0,1))

        # Префикс и суффикс (K,2,2) для выбранной позиции
        if pos == -1:
            pref, suff = I_K, M_tot
        elif pos == N:
            pref, suff = M_tot, I_K
        else:
            pref = np.transpose(stack.prefix[pos], (2,0,1))
            suff = np.transpose(stack.suffix[pos], (2,0,1))

        # Полная вариация матрицы: dM_tot = pref ⋅ dM_layer ⋅ suff  (K,2,2)
        dM_tot = np.einsum("kij,kjl,klm->kim", pref, dM_layer, suff)

        # Производные амплитуд: dr, dt (вектор по λ)
        dr, dt = _dr_dt_from_dM_vec(M_tot, q_in, q_sub, dM_tot)

        # Линейные поправки на толщину e
        r_new = stack.r + d_eps * dr
        t_new = stack.t + d_eps * dt
        # Энергетические коэффициенты до 1-го порядка
        R_new = np.abs(r_new)**2
        T_new = alpha * (np.abs(t_new)**2)

        MF_new = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, r_new, t_new, R_new, T_new)
        dmf = float(base_MF - MF_new)

        letters_best[k] = ins
        dmf_best[k] = dmf

    return np.array(positions), letters_best, dmf_best


# ---------------------------
# Needle-cycle
# ---------------------------
def needle_cycle(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    nH_values: np.ndarray,
    nL_values: np.ndarray,
    n_inc_values: np.ndarray,
    n_sub_values: np.ndarray,
    q_in: np.ndarray,
    q_sub: np.ndarray,
    qH: np.ndarray,
    qL: np.ndarray,
    kH: np.ndarray,
    kL: np.ndarray,
    pol: Literal["s","p"],
    theta_inc: float,
    cos_theta_in_H_layers: np.ndarray,
    cos_theta_in_L_layers: np.ndarray,
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

    t0 = None
    if log_timing:
        import time as _time
        t0 = _time.perf_counter()

    current = stack
    history: List[Dict[str, Any]] = []

    for step in range(max_steps):
        mf_before = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, current.r, current.t, current.R, current.T)

        t0 = time.perf_counter()
        # 1) Аналитическая P-карта (по Тихонравову)
        positions, letters_best, dmf_best = analytic_excitation_map(
            current, q_in, q_sub, qH, qL, kH, kL,
            nH_values, nL_values, n_inc_values,
            wavelengths, targets, pol, theta_inc, d_eps
        )
        t1 = time.perf_counter()
        print("time pmap: "+str(t1-t0))

        print("positions: "+str(positions))
        print("letters_best: "+str(letters_best))
        print("dmf_best: "+str(dmf_best))

        # Лучшая точка вставки — по максимуму выигрыша
        idx = int(np.argmax(dmf_best))
        if not np.isfinite(dmf_best[idx]) or dmf_best[idx] <= 0.0:
            # нет предсказанного улучшения
            break

        pos = int(positions[idx])
        ins = str(letters_best[idx])

        # 2) Реальная вставка слоя толщиной d_init и пересборка стека
        letters = [L.litera for L in current.layers]
        th = np.array([L.d for L in current.layers], dtype=float)

        pos = 6
        ins = "H"        

        if pos == -1:                      # перед первым слоем
            letters_new = [ins] + letters
            th_new = np.insert(th, 0, d_init)
        elif pos == len(letters):          # после последнего
            letters_new = letters + [ins]
            th_new = np.append(th, d_init)
        else:                              # середина слоя pos
            base_letter = letters[pos]
            d1 = 0.5 * th[pos]; d2 = th[pos] - d1
            letters_new = letters[:pos] + [base_letter, ins, base_letter] + letters[pos+1:]
            th_new = np.concatenate([th[:pos], [d1, d_init, d2], th[pos+1:]])

        current = make_stack_from_letters(
            letters_new, th_new,
            n_inc_values, n_sub_values, nH_values, nL_values,
            cos_theta_in_H_layers, cos_theta_in_L_layers,
            q_in, q_sub, qH, qL,
            wavelengths, len(wavelengths), pol
        )

        new_merit = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, current.r, current.t, current.R, current.T)
        print("new merit function: "+str(new_merit))

        # 3) Локальная доводка толщин (необязательна, но полезна)
        current, mf_after = coordinate_descent_thicknesses(
            current, wavelengths, targets,
            pol=pol, theta_inc=theta_inc,
            step_rel=coord_step_rel, min_step_rel=coord_min_step_rel,
            iters=coord_iters, d_min=d_min, d_max=d_max
        )
        history.append({"step": step, "MF": mf_after})

        # 4) Критерии останова
        if mf_before - mf_after < min_rel_improv * max(1.0, mf_before):
            break
        if layer_count(current) > max_layers:
            break
        if total_optical_thickness(current, wl_ref_for_tot) > max_tot_nmopt:
            break

    elapsed = None
    if log_timing and t0 is not None:
        import time as _time
        elapsed = _time.perf_counter() - t0

    return current, {"history": history, "elapsed": elapsed}
