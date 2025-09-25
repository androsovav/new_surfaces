# src/algorithms/optimizers.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Literal
from ..core.optics import Stack, make_M, rt_amplitudes, RT_coeffs
from ..core.merit import rms_merit, rms_merit_layers
from ..design.design import make_stack

def coordinate_descent_thicknesses(
    stack: Stack,
    wavelengths: np.ndarray,
    n_wavelengths: int,
    targets: dict,
    nH_values: np.ndarray,
    nL_values: np.ndarray,
    cos_theta_in_H_layers: np.ndarray,
    cos_theta_in_L_layers: np.ndarray,
    q_in: np.ndarray,
    q_sub: np.ndarray,
    qH: np.ndarray,
    qL: np.ndarray,
    pol: Literal["s","p"] = "s",
    theta_inc: float = 0.0,
    step_rel: float = 0.1,
    min_step_rel: float = 0.01,
    iters: int = 50,
    d_min: float | None = None,
    d_max: float | None = None,
) -> Tuple[Stack, float]:
    thickness = np.copy(stack.thickness)
    num_of_layers = len(thickness)

    # Текущие фазы/матрицы слоёв берём из входного стека
    phi = np.copy(stack.phi)                               # (N, K)
    M_layers = np.copy(stack.M_layers)                     # (2,2,N,K)

    def rebuild_prefix_suffix(M_layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prefix = np.empty((2,2,num_of_layers,n_wavelengths), dtype=complex)
        suffix = np.empty((2,2,num_of_layers,n_wavelengths), dtype=complex)
        left  = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,n_wavelengths))
        right = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,n_wavelengths))
        for i in range(num_of_layers):
            left = np.einsum('ijk,jlk->ilk', left, M_layers[:,:,i,:])
            prefix[:,:,i,:] = left
            right = np.einsum('ijk,jlk->ilk', M_layers[:,:,-(i+1),:], right)
            suffix[:,:,-(i+1),:] = right
        return prefix, suffix

    def update_prefix_suffix(prefix, suffix, M_layers, idx: int):
        """
        Обновляет только затронутые части prefix и suffix после изменения слоя idx.
        prefix, suffix: (2,2,N,K)
        M_layers: (2,2,N,K)
        idx: индекс изменённого слоя
        """

        N = M_layers.shape[2]
        K = M_layers.shape[3]

        # --- обновляем prefix начиная с idx ---
        if idx == 0:
            left = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))
        else:
            left = prefix[:, :, idx-1, :]

        for j in range(idx, N):
            left = np.einsum("ijk,jlk->ilk", left, M_layers[:, :, j, :])
            prefix[:, :, j, :] = left

        # --- обновляем suffix начиная с idx ---
        if idx == N-1:
            right = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))
        else:
            right = suffix[:, :, idx+1, :]

        for j in range(idx, -1, -1):
            right = np.einsum("ijk,jlk->ilk", M_layers[:, :, j, :], right)
            suffix[:, :, j, :] = right


    prefix, suffix = rebuild_prefix_suffix(M_layers)

    mf = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc,
                   stack.r, stack.t, stack.R, stack.T)

    step = float(step_rel)
    for _ in range(iters):
        improved = False

        # Кандидаты: «плюс» и «минус»
        mf_new_plus = mf_new_minus = None
        M_changed_plus = M_changed_minus = None

        for sgn in (-1, +1):
            # ограничим толщины, если заданы границы
            scale = (1.0 + sgn * step)
            if d_min is not None or d_max is not None:
                # слой-за-слоем ограничим масштаб, где это нарушает границы
                scale_vec = np.full(num_of_layers, scale, dtype=float)
                if d_min is not None:
                    scale_vec = np.where(thickness*scale_vec < d_min, d_min/np.maximum(thickness, 1e-30), scale_vec)
                if d_max is not None:
                    scale_vec = np.where(thickness*scale_vec > d_max, d_max/np.maximum(thickness, 1e-30), scale_vec)
            else:
                scale_vec = scale

            phi_changed = phi * scale_vec[:, None]         # (N,K)
            sphi_changed = np.sin(phi_changed)
            cphi_changed = np.cos(phi_changed)
            M_changed = make_M(sphi_changed, cphi_changed, stack.q, num_of_layers, n_wavelengths)

            # Собираем M_total для «замены одного слоя» (батч по всем слоям)
            M_total_changed = np.empty_like(M_changed)     # (2,2,N,K)

            if num_of_layers == 1:
                M_total_changed[:,: ,0,:] = M_changed[:,: ,0,:]
            else:
                # середина
                if num_of_layers >= 3:
                    M_total_changed[:,:,1:-1,:] = np.einsum(
                        'abnw, bcnw, cdnw -> adnw',
                        prefix[:,:,:-2,:],             # до i-1
                        M_changed[:,:,1:-1,:],         # i
                        suffix[:,:,2:,:]               # после i+1
                    )
                # края
                M_total_changed[:,:,0,:]  = np.einsum('abw,bcw->acw', M_changed[:,:,0,:],  suffix[:,:,1,:])
                M_total_changed[:,:,-1,:] = np.einsum('abw,bcw->acw', prefix[:,:,-2,:],    M_changed[:,:,-1,:])

            r, t = rt_amplitudes(M_total_changed, q_in, q_sub)
            R, T = RT_coeffs(r, t, q_in, q_sub)

            mf_layers = rms_merit_layers(q_in, q_sub, wavelengths, targets, pol, theta_inc, r, t, R, T)

            if sgn == +1:
                mf_new_plus, M_changed_plus = mf_layers, M_changed
            else:
                mf_new_minus, M_changed_minus = mf_layers, M_changed

        # Выбираем лучшее направление и слой
        if np.min(mf_new_plus) < np.min(mf_new_minus):
            mf_new = mf_new_plus
            chosen_sgn = +1
            M_changed_best = M_changed_plus
        else:
            mf_new = mf_new_minus
            chosen_sgn = -1
            M_changed_best = M_changed_minus

        best_idx = int(np.argmin(mf_new))

        if mf_new[best_idx] < mf:
            # ВАЖНО: применяем правильный знак шага
            improved = True
            mf = float(mf_new[best_idx])

            # Обновляем толщину и фазу только лучшего слоя
            scale_best = 1.0 + chosen_sgn * step
            if d_min is not None:
                scale_best = max(scale_best, (d_min / max(thickness[best_idx], 1e-30)))
            if d_max is not None:
                scale_best = min(scale_best, (d_max / max(thickness[best_idx], 1e-30)))

            thickness[best_idx] *= scale_best
            phi[best_idx, :]   *= scale_best

            M_layers[:,:,best_idx,:] = M_changed_best[:,:,best_idx,:]
            update_prefix_suffix(prefix, suffix, M_layers, best_idx)


        if not improved:
            step *= 0.95
            if step < min_step_rel:
                break

    # Пересобираем финальный стек и мерит честно
    res = make_stack(stack.start_flag, thickness, nH_values, nL_values,
                     cos_theta_in_H_layers, cos_theta_in_L_layers,
                     q_in, q_sub, qH, qL, wavelengths, n_wavelengths,
                     calculate_prefix_and_suffix_for_needle=True)
    res_mf = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, res.r, res.t, res.R, res.T)
    return res, res_mf