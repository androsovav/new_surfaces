# src/algorithms/needle.py
from __future__ import annotations
import numpy as np
import time
from typing import List, Tuple, Literal, Dict, Any

from ..core.optics import (
    Stack, phi_parameter, make_M, rt_amplitudes, RT_coeffs
)
from ..core.merit import rms_merit, rms_merit_layers
from ..algorithms.optimizers import coordinate_descent_thicknesses
from ..core.metrics import total_optical_thickness
from ..design.design import make_stack_from_letters, make_stack, add_prefix_and_suffix_to_stack

# ---------------------------
# Вспомогательная сборка стека по произвольной H/L-последовательности
# (аналог design.make_stack, но принимает letters[])
# ---------------------------


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
    m11 = M[0, 0, :]; m12 = M[0, 1, :]
    m21 = M[1, 0, :]; m22 = M[1, 1, :]

    dm11 = dM[0, 0, :]; dm12 = dM[0, 1, :]
    dm21 = dM[1, 0, :]; dm22 = dM[1, 1, :]

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
    dM_in_H_layer: np.ndarray,
    dM_in_L_layer: np.ndarray,
    wavelengths: np.ndarray,
    targets: dict,
    pol: Literal["s","p"],
    theta_inc: float,
    d_eps: float,
    num_of_layers: int,
    n_wavelengths: int
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
    num_of_layers = len(stack.thickness)
    positions = np.array(range(num_of_layers))

    mf_best = np.full(len(positions), -np.inf, dtype=float)
    I_K = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, len(wavelengths)))

    M_total = stack.M

    alpha = np.real(q_sub/q_in)           # для T = alpha * |t|^2

    dM = np.empty((2, 2, num_of_layers, n_wavelengths), dtype=np.ndarray)
    if stack.start_flag == "H":
        dM[:, :, 0::2, :] = dM_in_H_layer[:, :, None, :]
        dM[:, :, 1::2, :] = dM_in_L_layer[:, :, None, :]
    else:
        dM[:, :, 0::2, :] = dM_in_L_layer[:, :, None, :]
        dM[:, :, 1::2, :] = dM_in_H_layer[:, :, None, :]

    dM_total_changed = np.einsum(
        'abnw, bcnw, cdnw -> adnw',
        stack.prefix,
        dM,
        stack.suffix
    )

    dr, dt = _dr_dt_from_dM_vec(M_total, q_in, q_sub, dM_total_changed)

    print("dr")
    print(np.shape(dr))
    print("dt")
    print(np.shape(dt))

    # Линейные поправки на толщину e
    r_new = stack.r + d_eps * dr
    t_new = stack.t + d_eps * dt

    print("r_new")
    print(np.shape(r_new))
    print("t_new")
    print(np.shape(t_new))
    # Энергетические коэффициенты до 1-го порядка
    R_new = np.abs(r_new)**2
    T_new = alpha * (np.abs(t_new)**2)

    print("R_new")
    print(np.shape(R_new))
    print("T_new")
    print(np.shape(T_new))
    print("DEBUG shapes:", np.shape(r_new), np.shape(t_new), np.shape(R_new), np.shape(T_new))
    MF_new = rms_merit_layers(q_in, q_sub, wavelengths, targets, pol, theta_inc, r_new, t_new, R_new, T_new)
    print(np.shape(MF_new))
    quit()

    for pos in positions:
        # Выбор комплементарной «иглы» (H↔L) в этой позиции
        if pos == -1:
            ins = "L" if stack.start_flag == "H" else "H"
        elif pos == num_of_layers:
            if num_of_layers % 2 == 1:
                ins = "L" if stack.start_flag == "H" else "H"
            else:
                ins = "H" if stack.start_flag == "H" else "L"
        else:
            if pos % 2 == 0:
                ins = "L" if stack.start_flag == "H" else "H"
            else:
                ins = "H" if stack.start_flag == "H" else "L"

        print(ins)

        if ins == "H":
            dM_layer = dM_in_H_layer
        else:
            dM_layer = dM_in_L_layer

        # Префикс и суффикс (K,2,2) для выбранной позиции
        if pos == -1:
            pref, suff = I_K, M_total
        elif pos == num_of_layers:
            pref, suff = M_total, I_K
        else:
            pref = stack.prefix[:, :, pos, :]
            suff = stack.suffix[:, :, pos, :]

        # Полная вариация матрицы: dM_tot = pref ⋅ dM_layer ⋅ suff  (K,2,2)
        dM_tot = np.einsum("kij,kjl,klm->kim", pref, dM_layer, suff)

        # Производные амплитуд: dr, dt (вектор по λ)
        dr, dt = _dr_dt_from_dM_vec(M_total, q_in, q_sub, dM_tot)

        # Линейные поправки на толщину e
        r_new = stack.r + d_eps * dr
        t_new = stack.t + d_eps * dt
        # Энергетические коэффициенты до 1-го порядка
        R_new = np.abs(r_new)**2
        T_new = alpha * (np.abs(t_new)**2)

        MF_new = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, r_new, t_new, R_new, T_new)

        mf_best[pos] = MF_new
    
    print("positions")
    print(positions)
    print("mf_best")
    print(mf_best)
    quit()
    return positions, mf_best


# ---------------------------
# Needle-cycle
# ---------------------------
def needle_cycle(
    stack: Stack,
    wavelengths: np.ndarray,
    n_wavelengths: int,
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
    dM_in_H_layer: np.ndarray,
    dM_in_L_layer: np.ndarray,
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
        num_of_layers = len(current.thickness)

        # 1) Аналитическая P-карта (по Тихонравову)
        positions, mf_best = analytic_excitation_map(
            current, q_in, q_sub, qH, qL, kH, kL, dM_in_H_layer, dM_in_L_layer,
            wavelengths, targets, pol, theta_inc, d_eps, num_of_layers, n_wavelengths
        )

        # Лучшая точка вставки — по максимуму выигрыша
        idx = int(np.argmin(mf_best))
        if not np.isfinite(mf_best[idx]) or (mf_before - mf_best[idx]) <= 0.0:
            # нет предсказанного улучшения
            break

        pos = int(positions[idx])

        # 2) Реальная вставка слоя толщиной d_init и пересборка стека
        th = current.thickness

        start_flag = current.start_flag

        if pos == -1:                      # перед первым слоем
            if start_flag == "H":
                start_flag = "L"
            else:
                start_flag = "H"
            th_new = np.insert(th, 0, d_init)
        elif pos == len(th):          # после последнего
            th_new = np.append(th, d_init)
        else:                              # середина слоя pos
            d1 = 0.5 * th[pos]; d2 = th[pos] - d1
            th_new = np.concatenate([th[:pos], [d1, d_init, d2], th[pos+1:]])

        current = make_stack(start_flag, th_new, nH_values, nL_values, cos_theta_in_H_layers, cos_theta_in_L_layers,
                             q_in, q_sub, qH, qL, wavelengths, n_wavelengths, calculate_prefix_and_suffix_for_needle=False)

        new_merit = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, current.r, current.t, current.R, current.T)
        print("new merit function: "+str(new_merit))

        # 3) Локальная доводка толщин (необязательна, но полезна)
        current, mf_after = coordinate_descent_thicknesses(
            current, wavelengths, n_wavelengths, targets,
            nH_values=nH_values,
            nL_values=nL_values,
            cos_theta_in_H_layers=cos_theta_in_H_layers,
            cos_theta_in_L_layers=cos_theta_in_L_layers,
            q_in=q_in,
            q_sub=q_sub,
            qH=qH,
            qL=qL,
            pol=pol,
            theta_inc=theta_inc,
            step_rel=coord_step_rel,
            min_step_rel=coord_min_step_rel,
            iters=coord_iters,
            d_min=d_min,
            d_max=d_max,
        )
        current = add_prefix_and_suffix_to_stack(current, n_wavelengths)

        history.append({"step": step, "MF": mf_after})

        # 4) Критерии останова
        if mf_before - mf_after < min_rel_improv * max(1.0, mf_before):
            break
        if len(current.thickness) > max_layers:
            break
        if total_optical_thickness(current, wl_ref_for_tot, nH_values, nL_values, wavelengths) > max_tot_nmopt:
            break

    elapsed = None
    if log_timing and t0 is not None:
        import time as _time
        elapsed = _time.perf_counter() - t0

    return current, {"history": history, "elapsed": elapsed}
