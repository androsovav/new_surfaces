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


import numpy as np

def _dr_dt_from_dM_vec(
    M: np.ndarray,
    q_in: np.ndarray,
    q_sub: np.ndarray,
    dM: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Производные r и t по вариациям dM для всех слоёв.

    Параметры
    ---------
    M : np.ndarray, shape (2,2,K)
        Полная матрица стека для каждой длины волны.
    q_in : np.ndarray, shape (K,)
        Волновое число (адмиттанс) падающей среды.
    q_sub : np.ndarray, shape (K,)
        Волновое число (адмиттанс) подложки.
    dM : np.ndarray, shape (2,2,N,K)
        Вариации полной матрицы по толщине для каждого слоя.

    Возврат
    -------
    dr, dt : np.ndarray, shape (N,K)
        Поправки к r и t для каждой позиции вставки и каждой длины волны.
    """
    # элементы матрицы M (для всех K)
    M11, M12 = M[0,0,:], M[0,1,:]
    M21, M22 = M[1,0,:], M[1,1,:]

    # корректные формулы
    denom = M11*q_sub + M12 + M21*q_in*q_sub + M22*q_in
    num_r = M11*q_sub + M12 - (M21*q_in*q_sub + M22*q_in)
    num_t = 2*q_in

    dM11, dM12 = dM[0,0,:,:], dM[0,1,:,:]
    dM21, dM22 = dM[1,0,:,:], dM[1,1,:,:]

    ddenom = dM11*q_sub[None,:] + dM12 + dM21*(q_in[None,:]*q_sub[None,:]) + dM22*q_in[None,:]
    dnum_r = dM11*q_sub[None,:] + dM12 - (dM21*(q_in[None,:]*q_sub[None,:]) + dM22*q_in[None,:])

    dr = (dnum_r*denom[None,:] - num_r[None,:]*ddenom) / (denom[None,:]**2)
    dt = -(num_t[None,:] * ddenom) / (denom[None,:]**2)
    return dr, dt


# ---------------------------
# Аналитическая P-карта (по Тихонравову)
# ---------------------------
def analytic_excitation_map(
    stack: Stack,
    q_in: np.ndarray, q_sub: np.ndarray,
    alpha: np.ndarray,
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
    num_of_layers = len(stack.thickness)
    positions = np.array(range(num_of_layers))
    MF_old = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, stack.r, stack.t, stack.R, stack.T)
    print("MF in Pmap: "+str(MF_old))

    mf_best = np.full(len(positions), -np.inf, dtype=float)
    I_K = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, len(wavelengths)))

    M_total = stack.M

    dM = np.empty((2, 2, num_of_layers, n_wavelengths), dtype=complex)
    if stack.start_flag == "H":
        dM[:, :, 0::2, :] = dM_in_L_layer[:, :, None, :]
        dM[:, :, 1::2, :] = dM_in_H_layer[:, :, None, :]
    else:
        dM[:, :, 0::2, :] = dM_in_H_layer[:, :, None, :]
        dM[:, :, 1::2, :] = dM_in_L_layer[:, :, None, :]

    dM_total_changed = np.einsum(
        'abnw, bcnw, cdnw -> adnw',
        stack.prefix,
        dM,
        stack.suffix
    )

    # кандидат -1: prefix = I, suffix = M_total
    # (2,2,K) = dM_layer_first (2,2,K) · M_total (2,2,K)
    dM_minus1 = np.einsum('abk, bck -> ack', dM[:, :, 0, :], M_total)
    dM_minus1 = dM_minus1[:, :, None, :]  # (2,2,1,K)

    # кандидат N: prefix = M_total, suffix = I
    # (2,2,K) = M_total (2,2,K) · dM_layer_last (2,2,K)
    dM_plusN = np.einsum('abk, bck -> ack', M_total, dM[:, :, -1, :])
    dM_plusN = dM_plusN[:, :, None, :]  # (2,2,1,K)

    # объединяем все кандидаты
    dM_all = np.concatenate([dM_minus1, dM_total_changed, dM_plusN], axis=2)  # (2,2,N+2,K)

    dr, dt = _dr_dt_from_dM_vec(M_total, q_in, q_sub, dM_all)

    # Линейные поправки на толщину e
    r_new = stack.r[None, :] + d_eps * dr
    t_new = stack.t[None, :] + d_eps * dt
    # Энергетические коэффициенты до 1-го порядка
    R_new = stack.R[None, :] + 2.0*np.real(np.conj(stack.r)[None, :] * dr) * d_eps
    T_new = stack.T[None, :] + 2.0*alpha[None, :] * np.real(np.conj(stack.t)[None, :] * dt) * d_eps


    MF_new = rms_merit_layers(q_in, q_sub, wavelengths, targets, pol, theta_inc, r_new, t_new, R_new, T_new)

    return MF_old-MF_new


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
    alpha: np.ndarray,
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
        dmf = analytic_excitation_map(
            current, q_in, q_sub, alpha, qH, qL, kH, kL, dM_in_H_layer, dM_in_L_layer,
            wavelengths, targets, pol, theta_inc, d_eps, num_of_layers, n_wavelengths
        )

        # Лучшая точка вставки — по максимуму выигрыша
        idx = int(np.argmax(dmf))
        if not np.isfinite(dmf[idx]) or dmf[idx] <= 0.0:
            print("Нет хорошего места для иглы")
            break

        # 2) Реальная вставка слоя толщиной d_init и сборка КАНДИДАТА
        th = current.thickness
        start_flag = current.start_flag
        pos = idx - 1

        if pos == -1:                      # перед первым слоем
            start_flag = "L" if start_flag == "H" else "H"
            th_new = np.insert(th, 0, d_init)
        elif pos == len(th):               # после последнего
            th_new = np.append(th, d_init)
        else:                              # середина слоя pos
            d1 = 0.5 * th[pos]; d2 = th[pos] - d1
            th_new = np.concatenate([th[:pos], [d1, d_init, d2], th[pos+1:]])

        candidate = make_stack(
            start_flag, th_new, nH_values, nL_values,
            cos_theta_in_H_layers, cos_theta_in_L_layers,
            q_in, q_sub, qH, qL, wavelengths, n_wavelengths,
            calculate_prefix_and_suffix_for_needle=False
        )

        new_merit = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc,
                            candidate.r, candidate.t, candidate.R, candidate.T)
        print("MF after needle: " + str(new_merit))

        # 3) Локальная доводка толщин на КАНДИДАТЕ
        cand_opt, mf_after = coordinate_descent_thicknesses(
            candidate, wavelengths, n_wavelengths, targets,
            nH_values=nH_values, nL_values=nL_values,
            cos_theta_in_H_layers=cos_theta_in_H_layers,
            cos_theta_in_L_layers=cos_theta_in_L_layers,
            q_in=q_in, q_sub=q_sub, qH=qH, qL=qL,
            pol=pol, theta_inc=theta_inc,
            step_rel=coord_step_rel, min_step_rel=coord_min_step_rel,
            iters=coord_iters, d_min=d_min, d_max=d_max
        )
        cand_opt = add_prefix_and_suffix_to_stack(cand_opt, n_wavelengths)
        print("MF after coordinate: " + str(mf_after))

        # пересчитаем честно (на cand_opt)
        mf_after = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc,
                            cand_opt.r, cand_opt.t, cand_opt.R, cand_opt.T)
        print("real MF after coordinate: " + str(mf_after))

        # 4) Приём/откат
        tol = min_rel_improv * max(1.0, mf_before)
        delta = mf_before - mf_after

        if (delta > 0.0) and (delta >= tol):
            # ПРИНИМАЕМ кандидата
            current = cand_opt
            history.append({"step": step, "MF": mf_after})
        else:
            # ОТКАТ — оставляем прежний current
            print(f"Откат вставки: улучшения нет (Δ={delta:.3e} < tol={tol:.3e})")
            break

        # 4) Критерии останова
        if mf_before - mf_after < min_rel_improv * max(1.0, mf_before):
            print("Слишком слабое улучшение")
            break
        if len(current.thickness) > max_layers:
            print("Слишком много слоев")
            break
        if total_optical_thickness(current, wl_ref_for_tot, nH_values, nL_values, wavelengths) > max_tot_nmopt:
            print("Слишком большая оптическая толщина")
            break

    elapsed = None
    if log_timing and t0 is not None:
        import time as _time
        elapsed = _time.perf_counter() - t0

    return current, {"history": history, "elapsed": elapsed}
