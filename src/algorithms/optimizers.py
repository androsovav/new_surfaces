# src/algorithms/optimizers.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from ..core.optics import Stack, make_M
from ..core.merit import rms_merit, rms_merit_layers
from ..design.design import make_stack

def _rebuild_prefix_suffix_all_pols(M_layers_by_pol: dict[str, np.ndarray],
                                    n_wavelengths: int) -> tuple[dict, dict]:
    """
    Полная пересборка prefix/suffix для каждой поляризации.
    prefix[pol][:,:,i,:] = M(0..i) (накопительное слева)
    suffix[pol][:,:,i,:] = M(i..N-1) (накопительное справа)
    """
    prefix, suffix = {}, {}
    any_pol = next(iter(M_layers_by_pol.keys()))
    N = M_layers_by_pol[any_pol].shape[2]
    K = n_wavelengths

    I = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))

    for pol, M_layers in M_layers_by_pol.items():
        prefix_pol = np.empty((2, 2, N, K), dtype=complex)
        suffix_pol = np.empty((2, 2, N, K), dtype=complex)

        left = I.copy()
        right = I.copy()

        for i in range(N):
            left = np.einsum("ijk,jlk->ilk", left, M_layers[:, :, i, :])
            prefix_pol[:, :, i, :] = left

            j = N - 1 - i
            right = np.einsum("ijk,jlk->ilk", M_layers[:, :, j, :], right)
            suffix_pol[:, :, j, :] = right

        prefix[pol] = prefix_pol
        suffix[pol] = suffix_pol

    return prefix, suffix


def _update_prefix_suffix_from_idx(prefix_by_pol: dict, suffix_by_pol: dict,
                                   M_layers_by_pol: dict, idx: int) -> None:
    """
    Локальное обновление prefix/suffix начиная с изменённого слоя idx.
    """
    any_pol = next(iter(M_layers_by_pol.keys()))
    N = M_layers_by_pol[any_pol].shape[2]
    K = M_layers_by_pol[any_pol].shape[3]

    I = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))

    for pol, M_layers in M_layers_by_pol.items():
        prefix = prefix_by_pol[pol]
        suffix = suffix_by_pol[pol]

        # --- обновляем prefix c idx до конца ---
        if idx == 0:
            left = I.copy()
        else:
            # prefix[:,:,idx-1,:] уже содержит M(0..idx-1)
            left = prefix[:, :, idx - 1, :]

        for j in range(idx, N):
            left = np.einsum("ijk,jlk->ilk", left, M_layers[:, :, j, :])
            prefix[:, :, j, :] = left

        # --- обновляем suffix от idx до начала ---
        if idx == N - 1:
            right = I.copy()
        else:
            # suffix[:,:,idx+1,:] уже содержит M(idx+1..N-1)
            right = suffix[:, :, idx + 1, :]

        for j in range(idx, -1, -1):
            right = np.einsum("ijk,jlk->ilk", M_layers[:, :, j, :], right)
            suffix[:, :, j, :] = right


def _rt_batch_for_layers(constants: dict, pol: str, M_total_changed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Считает r,t для батча полных матриц стека по всем слоям:
      M_total_changed: (2,2,N,K) -> r,t формы (N,K)
    """
    q_in = constants["q_in"][pol]    # (K,)
    q_sub = constants["q_sub"][pol]  # (K,)

    A = M_total_changed[0, 0, :, :]  # (N,K)
    B = M_total_changed[0, 1, :, :]
    C = M_total_changed[1, 0, :, :]
    D = M_total_changed[1, 1, :, :]

    X = A + B * q_sub[None, :]
    Y = C + D * q_sub[None, :]
    denom = X * q_in[None, :] + Y
    r = (X * q_in[None, :] - Y) / denom
    t = (2 * q_in[None, :]) / denom
    return r, t


def coordinate_descent_thicknesses(constants: dict, stack: Stack) -> Tuple[Stack, float]:
    """
    Координатный спуск по толщине слоёв под новую архитектуру:
      - все поля r/t/R/T, q, M_layers, prefix/suffix — словари по поляризациям
      - rms_merit / rms_merit_layers — работают с такими же словарями
    Возвращает (новый стек, mf).
    """
    thickness = np.copy(stack.thickness)
    N = len(thickness)
    K = constants["n_wavelengths"]

    # текущее φ и M_layers берём из стека (общая φ, разные q по pol)
    phi = np.copy(stack.phi)  # (N,K)
    M_layers_by_pol = {pol: np.copy(M) for pol, M in stack.M_layers.items()}

    # построим prefix/suffix (без half-слоёв: классическая накопительная схема)
    prefix_by_pol, suffix_by_pol = _rebuild_prefix_suffix_all_pols(M_layers_by_pol, K)

    # стартовая MF
    mf = rms_merit(constants, stack.r, stack.t, stack.R, stack.T)

    step = float(constants["step_rel"])
    for _ in range(int(constants["iters"])):
        improved = False

        mf_new_plus = mf_new_minus = None
        M_changed_plus = M_changed_minus = None

        for sgn in (-1, +1):
            # масштаб по слоям с учётом границ
            scale = (1.0 + sgn * step)
            if constants.get("d_min") is not None or constants.get("d_max") is not None:
                scale_vec = np.full(N, scale, dtype=float)
                if constants.get("d_min") is not None:
                    scale_vec = np.where(thickness * scale_vec < constants["d_min"],
                                         constants["d_min"] / np.maximum(thickness, 1e-30),
                                         scale_vec)
                if constants.get("d_max") is not None:
                    scale_vec = np.where(thickness * scale_vec > constants["d_max"],
                                         constants["d_max"] / np.maximum(thickness, 1e-30),
                                         scale_vec)
            else:
                scale_vec = scale

            # обновлённые фазы
            phi_changed = phi * scale_vec[:, None]          # (N,K)
            sphi_changed = np.sin(phi_changed)
            cphi_changed = np.cos(phi_changed)

            # пересчёт M_layers для КАЖДОЙ поляризации
            M_changed_by_pol = {}
            for pol in M_layers_by_pol.keys():
                q_pol = stack.q[pol]                        # (N,K)
                M_changed_by_pol[pol] = make_M(sphi_changed, cphi_changed, q_pol, N, K)  # (2,2,N,K)

            # соберём полные матрицы стека для «замены одного слоя» (батч по всем слоям)
            r_dict, t_dict, R_dict, T_dict = {}, {}, {}, {}

            for pol, M_changed in M_changed_by_pol.items():
                prefix = prefix_by_pol[pol]
                suffix = suffix_by_pol[pol]

                M_total_changed = np.empty_like(M_changed)  # (2,2,N,K)

                if N == 1:
                    M_total_changed[:, :, 0, :] = M_changed[:, :, 0, :]
                else:
                    # середина: prefix(0..i-1) * M_changed(i) * suffix(i+1..N-1)
                    if N >= 3:
                        M_total_changed[:, :, 1:-1, :] = np.einsum(
                            "abnk,bcnk,cdnk->adnk",
                            prefix[:, :, :-2, :],
                            M_changed[:, :, 1:-1, :],
                            suffix[:, :, 2:, :]
                        )
                    # края
                    M_total_changed[:, :, 0, :] = np.einsum("abk,bck->ack", M_changed[:, :, 0, :],  suffix[:, :, 1, :])
                    M_total_changed[:, :, -1, :] = np.einsum("abk,bck->ack", prefix[:, :, -2, :],   M_changed[:, :, -1, :])

                # r,t,R,T для всех i (батч)
                r_pol, t_pol = _rt_batch_for_layers(constants, pol, M_total_changed)   # (N,K)
                alpha = constants["alpha"][pol]                                        # (K,)
                R_pol = np.abs(r_pol) ** 2
                T_pol = (np.real(constants["q_sub"][pol]) / np.real(constants["q_in"][pol]))[None, :] * (np.abs(t_pol) ** 2)

                r_dict[pol] = r_pol
                t_dict[pol] = t_pol
                R_dict[pol] = R_pol
                T_dict[pol] = T_pol

            # мерит по слоям для этого направления
            mf_layers = rms_merit_layers(constants, r_dict, t_dict, R_dict, T_dict)  # (N,)

            if sgn == +1:
                mf_new_plus, M_changed_plus = mf_layers, M_changed_by_pol
            else:
                mf_new_minus, M_changed_minus = mf_layers, M_changed_by_pol

        # выбираем лучшее направление и слой
        if np.min(mf_new_plus) < np.min(mf_new_minus):
            mf_new = mf_new_plus
            chosen_sgn = +1
            M_changed_best_by_pol = M_changed_plus
        else:
            mf_new = mf_new_minus
            chosen_sgn = -1
            M_changed_best_by_pol = M_changed_minus

        best_idx = int(np.argmin(mf_new))

        if mf_new[best_idx] < mf:
            improved = True
            mf = float(mf_new[best_idx])

            # применяем масштаб к лучшему слою с учётом границ
            scale_best = 1.0 + chosen_sgn * step
            if constants.get("d_min") is not None:
                scale_best = max(scale_best, (constants["d_min"] / max(thickness[best_idx], 1e-30)))
            if constants.get("d_max") is not None:
                scale_best = min(scale_best, (constants["d_max"] / max(thickness[best_idx], 1e-30)))

            thickness[best_idx] *= scale_best
            phi[best_idx, :]   *= scale_best

            # обновляем только изменённый слой в M_layers и локально чинём prefix/suffix
            for pol in M_layers_by_pol.keys():
                M_layers_by_pol[pol][:, :, best_idx, :] = M_changed_best_by_pol[pol][:, :, best_idx, :]
            _update_prefix_suffix_from_idx(prefix_by_pol, suffix_by_pol, M_layers_by_pol, best_idx)

        if not improved:
            step *= 0.9
            if step < float(constants["min_step_rel"]):
                break

    # финальная сборка стека и "честный" MF
    res = make_stack(constants, stack.start_flag, thickness, calculate_prefix_and_suffix_for_needle=True)
    res_mf = rms_merit(constants, res.r, res.t, res.R, res.T)
    return res, res_mf
