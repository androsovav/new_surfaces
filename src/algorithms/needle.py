# src/algorithms/needle.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List

from ..core.merit import rms_merit, rms_merit_layers
from ..core.metrics import total_optical_thickness
from ..design.design import make_stack, add_prefix_and_suffix_to_stack
from .optimizers import coordinate_descent_thicknesses

# ---------------------------
# Вспомогательная сборка стека по произвольной H/L-последовательности
# (аналог design.make_stack, но принимает letters[])
# ---------------------------


import numpy as np

def _dr_dt_from_dM_vec_pol(q_in, q_sub, M, dM_all):
    """
    Поляризационная версия: принимает q_in, q_sub (K,), M (2,2,K),
    dM_all (2,2,Nc,K) и возвращает dr, dt формы (Nc, K).
    """
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]          # (K,)
    X = A + B*q_sub                                      # (K,)
    Y = C + D*q_sub                                      # (K,)
    denom = X*q_in + Y                                   # (K,)
    num_r = X*q_in - Y                                   # (K,)
    num_t = 2*q_in                                       # (K,)

    dA, dB = dM_all[0,0,:,:], dM_all[0,1,:,:]            # (Nc,K)
    dC, dD = dM_all[1,0,:,:], dM_all[1,1,:,:]            # (Nc,K)
    dX = dA + dB*q_sub[None,:]                           # (Nc,K)
    dY = dC + dD*q_sub[None,:]                           # (Nc,K)

    ddenom = dX*q_in[None,:] + dY                        # (Nc,K)
    dnum_r = dX*q_in[None,:] - dY                        # (Nc,K)

    dr = (dnum_r*denom[None,:] - num_r[None,:]*ddenom) / (denom[None,:]**2)
    dt = -(num_t[None,:] * ddenom) / (denom[None,:]**2)
    return dr, dt



# ---------------------------
# Аналитическая P-карта (по Тихонравову)
# ---------------------------
def analytic_excitation_map(constants, stack) -> np.ndarray:
    """
    Возвращает вектор dMF для кандидатов вставки: [-1, 0..N-1, N],
    с учётом обеих поляризаций (если цели их задействуют).
    """
    N = len(stack.thickness)
    K = constants["n_wavelengths"]

    # Базовая MF для текущего стека (поляризационная RMS)
    MF_old = rms_merit(constants, stack.r, stack.t, stack.R, stack.T)

    # --- построение dM по позициям для КАЖДОЙ поляризации ---
    # 1) dM для «материала иглы» в каждой позиции слоя (2,2,N,K)
    #    материал чередуется в зависимости от start_flag
    dM_by_pol = {}
    for pol in constants["qH"].keys():
        dM = np.empty((2,2,N,K), dtype=complex)
        if stack.start_flag == "H":
            dM[:,:,0::2,:] = constants["dM_in_L_layer"][pol][:,:,None,:]
            dM[:,:,1::2,:] = constants["dM_in_H_layer"][pol][:,:,None,:]
        else:
            dM[:,:,0::2,:] = constants["dM_in_H_layer"][pol][:,:,None,:]
            dM[:,:,1::2,:] = constants["dM_in_L_layer"][pol][:,:,None,:]
        dM_by_pol[pol] = dM

    # --- через prefix/suffix собираем dM_total для вставки в середину каждого слоя ---
    # dM_total_changed: (2,2,N,K) на позициях 0..N-1
    dM_total_changed = {}
    for pol in dM_by_pol.keys():
        dM = dM_by_pol[pol]
        pref = stack.prefix[pol]     # (2,2,N,K): prefix(i) = M(0..i-1) * M_half(i)
        suff = stack.suffix[pol]     # (2,2,N,K): suffix(i) = M_half(i) * M(i+1..N-1)
        dM_tot = np.einsum('abnk, bcnk, cdnk -> adnk', pref, dM, suff)  # (2,2,N,K)
        dM_total_changed[pol] = dM_tot

    # --- крайние кандидаты: -1 (до первого) и N (после последнего) ---
    # dM_all_pol: (2,2,Nc,K) где Nc = N+2
    dM_all_pol = {}
    for pol in dM_by_pol.keys():
        Mtot = stack.M[pol]                       # (2,2,K)
        dM = dM_by_pol[pol]
        # вставка перед первым: dM(-1) * M_total
        dM_minus1 = np.einsum('abk, bck -> ack', dM[:,:,0,:], Mtot)[:, :, None, :]  # (2,2,1,K)
        # вставка после последнего: M_total * dM(N-1)
        dM_plusN  = np.einsum('abk, bck -> ack', Mtot, dM[:,:,-1,:])[:, :, None, :] # (2,2,1,K)
        # объединяем
        dM_all = np.concatenate([dM_minus1, dM_total_changed[pol], dM_plusN], axis=2)  # (2,2,N+2,K)
        dM_all_pol[pol] = dM_all

    # --- вариации r,t по каждой поляризации ---
    dr_dict, dt_dict = {}, {}
    for pol in dM_all_pol.keys():
        q_in  = constants["q_in"][pol]    # (K,)
        q_sub = constants["q_sub"][pol]   # (K,)
        Mtot  = stack.M[pol]              # (2,2,K)
        dM_all = dM_all_pol[pol]          # (2,2,N+2,K)
        dr, dt = _dr_dt_from_dM_vec_pol(q_in, q_sub, Mtot, dM_all)  # (N+2,K)
        dr_dict[pol], dt_dict[pol] = dr, dt

    # --- линейные поправки r,t и энергетика R,T (1-й порядок по d_eps) ---
    r_new, t_new, R_new, T_new = {}, {}, {}, {}
    for pol in dr_dict.keys():
        dr, dt = dr_dict[pol], dt_dict[pol]      # (N+2,K)
        r0 = stack.r[pol][None, :]               # (1,K)
        t0 = stack.t[pol][None, :]
        r_new[pol] = r0 + constants["d_eps"] * dr
        t_new[pol] = t0 + constants["d_eps"] * dt

        # Первый порядок для R,T:
        # R_new ≈ R0 + 2 Re(conj(r0) * dr) * d_eps
        # T_new ≈ T0 + 2 alpha Re(conj(t0) * dt) * d_eps
        R0 = stack.R[pol][None, :]
        T0 = stack.T[pol][None, :]
        alpha = constants["alpha"][pol][None, :]  # (1,K)
        R_new[pol] = R0 + 2.0 * np.real(np.conj(r0) * dr) * constants["d_eps"]
        T_new[pol] = T0 + 2.0 * alpha * np.real(np.conj(t0) * dt) * constants["d_eps"]

    # --- оценка выигрыша по RMS для всех кандидатов ---
    MF_new = rms_merit_layers(constants, r_new, t_new, R_new, T_new)  # (N+2,)
    dmf = MF_old - MF_new
    return dmf


# ---------------------------
# Needle-cycle
# ---------------------------
def needle_cycle(constants: dict, stack, /) -> Tuple[Any, Dict[str, Any]]:
    """
    Итеративный цикл метода «иглы» (по Тихонравову) под новую архитектуру:
      - все величины r/t/R/T, M/prefix/suffix/q — это словари {"s": ..., "p": ...}
      - цели считываются из constants["targets"] (по поляризациям и глобальные)
    Возвращает (лучший_стек, информация_о_шаге).
    """
    # тайм-логирование
    t0 = None
    if constants.get("log_timing", False):
        import time as _time
        t0 = _time.perf_counter()

    current = stack
    history: List[Dict[str, Any]] = []

    for step in range(int(constants["max_steps"])):
        # 0) базовая мерит-функция
        mf_before = rms_merit(constants, current.r, current.t, current.R, current.T)

        # 1) Аналитическая P-карта: получаем ожидаемые выигрыши по позициям
        dmf = analytic_excitation_map(constants, current)  # shape: (N+2,)

        # лучшая точка вставки по максимуму выигрыша
        idx = int(np.argmax(dmf))
        if not np.isfinite(dmf[idx]) or dmf[idx] <= 0.0:
            # нет разумного места для вставки
            print("Нет хорошего места для иглы")
            break

        # индекс реальной позиции вставки в терминологии «между слоями»
        # наша P-карта возвращает кандидатов: [-1, 0..N-1, N] -> pos = idx-1
        pos = idx - 1
        th = current.thickness
        start_flag = current.start_flag

        # 2) Реальная вставка иглы толщиной d_init
        if pos == -1:
            # перед первым слоем — меняем стартовый флаг (чтоб сохранилась чередуемость материалов)
            start_flag = "L" if start_flag == "H" else "H"
            th_new = np.insert(th, 0, constants["d_init"])
        elif pos == len(th):
            # после последнего слоя
            th_new = np.append(th, constants["d_init"])
        else:
            # в середину слоя pos: разрезаем слой пополам и кладём иглу посередине
            d1 = 0.5 * th[pos]
            d2 = th[pos] - d1
            th_new = np.concatenate([th[:pos], [d1, constants["d_init"], d2], th[pos+1:]])

        # собираем кандидата без prefix/suffix (быстрее), затем при необходимости добавим
        candidate = make_stack(constants, start_flag, th_new, calculate_prefix_and_suffix_for_needle=False)

        # 3) Мерит кандидата «как есть»
        mf_after_needle = rms_merit(constants, candidate.r, candidate.t, candidate.R, candidate.T)
        print(f"MF after needle: {mf_after_needle}")

        # 4) Локальная доводка толщин на кандидате (будет отрефакторена под словари далее)
        #    На период миграции — пробуем, иначе идём без доводки.
        cand_opt = candidate
        mf_after = mf_after_needle
        try:
            cand_opt, mf_after = coordinate_descent_thicknesses(constants, candidate)
            # для корректной работы последующих шагов при методе иглы нам нужны prefix/suffix
            cand_opt = add_prefix_and_suffix_to_stack(cand_opt, constants["n_wavelengths"])
            print(f"MF after coordinate: {mf_after}")
            # «честно» пересчитываем MF — уже должно соответствовать словарной архитектуре
            mf_after = rms_merit(constants, cand_opt.r, cand_opt.t, cand_opt.R, cand_opt.T)
            print(f"real MF after coordinate: {mf_after}")
        except Exception as e:
            print(f"[coordinate_descent_thicknesses] временно пропускаем доводку: {e}")

        # 5) Критерий приёма/отката
        tol = float(constants["min_rel_improv"]) * max(1.0, mf_before)
        delta = mf_before - mf_after

        if (delta > 0.0) and (delta >= tol):
            # принимаем кандидата
            current = cand_opt
            history.append({"step": step, "MF": float(mf_after), "layers": len(current.thickness)})
        else:
            print(f"Откат вставки: улучшения нет (Δ={delta:.3e} < tol={tol:.3e})")
            break

        # 6) Остановы
        if (mf_before - mf_after) < float(constants["min_rel_improv"]) * max(1.0, mf_before):
            print("Слишком слабое улучшение")
            break
        if len(current.thickness) > int(constants["max_layers"]):
            print("Слишком много слоёв")
            break
        if total_optical_thickness(constants, current) > float(constants["max_tot_nmopt"]):
            print("Слишком большая оптическая толщина")
            break

    elapsed = None
    if constants.get("log_timing", False) and t0 is not None:
        import time as _time
        elapsed = _time.perf_counter() - t0

    return current, {"history": history, "elapsed": elapsed}
