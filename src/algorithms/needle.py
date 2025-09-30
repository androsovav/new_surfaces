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
from ..design.design import make_stack, add_prefix_and_suffix_to_stack

# ---------------------------
# Вспомогательная сборка стека по произвольной H/L-последовательности
# (аналог design.make_stack, но принимает letters[])
# ---------------------------


import numpy as np

def _dr_dt_from_dM_vec(constants, M, dM):
    # базовые обозначения
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]          # (K,)
    q_in, q_sub = constants["q_in"], constants["q_sub"]   # (K,)

    # r,t как в rt_amplitudes
    X = A + B*q_sub                  # (K,)
    Y = C + D*q_sub                  # (K,)
    denom = X*q_in + Y               # (K,)
    num_r = X*q_in - Y               # (K,)
    num_t = 2*q_in                   # (K,)

    # вариации элементов M_total
    dA, dB = dM[0,0,:,:], dM[0,1,:,:]   # (N,K)
    dC, dD = dM[1,0,:,:], dM[1,1,:,:]   # (N,K)
    dX = dA + dB*q_sub[None,:]          # (N,K)
    dY = dC + dD*q_sub[None,:]          # (N,K)

    ddenom = dX*q_in[None,:] + dY       # (N,K)
    dnum_r = dX*q_in[None,:] - dY       # (N,K)

    dr = (dnum_r*denom[None,:] - num_r[None,:]*ddenom) / (denom[None,:]**2)
    dt = -(num_t[None,:] * ddenom) / (denom[None,:]**2)
    return dr, dt



# ---------------------------
# Аналитическая P-карта (по Тихонравову)
# ---------------------------
def analytic_excitation_map(constants, stack: Stack) -> np.ndarray:
    num_of_layers = len(stack.thickness)
    positions = np.array(range(num_of_layers))
    MF_old = rms_merit(constants, stack.r, stack.t, stack.R, stack.T)
    print("MF in Pmap: "+str(MF_old))

    mf_best = np.full(len(positions), -np.inf, dtype=float)
    I_K = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, constants["n_wavelengths"]))

    M_total = stack.M

    dM = np.empty((2, 2, num_of_layers, constants["n_wavelengths"]), dtype=complex)
    if stack.start_flag == "H":
        dM[:, :, 0::2, :] = constants["dM_in_L_layer"][:, :, None, :]
        dM[:, :, 1::2, :] = constants["dM_in_H_layer"][:, :, None, :]
    else:
        dM[:, :, 0::2, :] = constants["dM_in_H_layer"][:, :, None, :]
        dM[:, :, 1::2, :] = constants["dM_in_L_layer"][:, :, None, :]

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

    dr, dt = _dr_dt_from_dM_vec(constants, M_total, dM_all)

    # Линейные поправки на толщину e
    r_new = stack.r[None, :] + constants["d_eps"] * dr
    t_new = stack.t[None, :] + constants["d_eps"] * dt
    # Энергетические коэффициенты до 1-го порядка
    R_new = stack.R[None, :] + 2.0*np.real(np.conj(stack.r)[None, :] * dr) * constants["d_eps"]
    T_new = stack.T[None, :] + 2.0*constants["alpha"][None, :] * np.real(np.conj(stack.t)[None, :] * dt) * constants["d_eps"]

    MF_new = rms_merit_layers(constants, r_new, t_new, R_new, T_new)

    dmf = MF_old-MF_new

    return dmf


# ---------------------------
# Needle-cycle
# ---------------------------
def needle_cycle(constants: dict, stack: Stack) -> Tuple[Stack, Dict[str, Any]]:

    t0 = None
    if constants["log_timing"]:
        import time as _time
        t0 = _time.perf_counter()

    current = stack
    history: List[Dict[str, Any]] = []

    for step in range(constants["max_steps"]):
        mf_before = rms_merit(constants, current.r, current.t, current.R, current.T)

        # 1) Аналитическая P-карта (по Тихонравову)
        dmf = analytic_excitation_map(constants, current)

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
            th_new = np.insert(th, 0, constants["d_init"])
        elif pos == len(th):               # после последнего
            th_new = np.append(th, constants["d_init"])
        else:                              # середина слоя pos
            d1 = 0.5 * th[pos]; d2 = th[pos] - d1
            th_new = np.concatenate([th[:pos], [d1, constants["d_init"], d2], th[pos+1:]])

        candidate = make_stack(constants, start_flag, th_new, calculate_prefix_and_suffix_for_needle=False)

        new_merit = rms_merit(constants, candidate.r, candidate.t, candidate.R, candidate.T)
        print("MF after needle: " + str(new_merit))

        # 3) Локальная доводка толщин на КАНДИДАТЕ
        cand_opt, mf_after = coordinate_descent_thicknesses(constants, candidate)
        cand_opt = add_prefix_and_suffix_to_stack(cand_opt, constants["n_wavelengths"])
        print("MF after coordinate: " + str(mf_after))

        # пересчитаем честно (на cand_opt)
        mf_after = rms_merit(constants, cand_opt.r, cand_opt.t, cand_opt.R, cand_opt.T)
        print("real MF after coordinate: " + str(mf_after))

        # 4) Приём/откат
        tol = constants["min_rel_improv"] * max(1.0, mf_before)
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
        if mf_before - mf_after < constants["min_rel_improv"] * max(1.0, mf_before):
            print("Слишком слабое улучшение")
            break
        if len(current.thickness) > constants["max_layers"]:
            print("Слишком много слоев")
            break
        if total_optical_thickness(constants, current) > constants["max_tot_nmopt"]:
            print("Слишком большая оптическая толщина")
            break

    elapsed = None
    if constants["log_timing"] and t0 is not None:
        import time as _time
        elapsed = _time.perf_counter() - t0

    return current, {"history": history, "elapsed": elapsed}
