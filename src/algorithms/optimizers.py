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
    """
    Координатный спуск по толщинам с постепенным уменьшением шага.
    """

    start_flag = stack.start_flag
    thickness = np.copy(stack.thickness)
    num_of_layers = len(stack.thickness)
    prefix = np.empty((2,2,num_of_layers,n_wavelengths), dtype=complex)
    suffix = np.empty((2,2,num_of_layers,n_wavelengths), dtype=complex)
    left  = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,n_wavelengths))   # (2,2,n_wavelength)
    right = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,n_wavelengths))   # (2,2,n_wavelength)

    phi = np.copy(stack.phi)

    for i in range(num_of_layers):
        left = np.einsum('ijk,jlk->ilk', left, stack.M_layers[:,:,i,:])
        right = np.einsum('ijk,jlk->ilk', stack.M_layers[:,:,-(i+1),:], right)
        prefix[:,:,i,:] = left
        suffix[:,:,-(i+1),:] = right

    mf = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc,
                   stack.r, stack.t, stack.R, stack.T)
    step = step_rel

    for _ in range(iters):
        improved = False
        for sgn in (+1, -1):
            phi_changed = phi * (1.0 + sgn * step)
            sphi_changed = np.sin(phi_changed)
            cphi_changed = np.cos(phi_changed)
            M_changed = make_M(sphi_changed, cphi_changed, stack.q, num_of_layers, n_wavelengths)
            M_total_changed = np.empty_like(M_changed)

            M_total_changed[:,:,1:-1,:] = np.einsum(
                'abnw, bcnw, cdnw -> adnw',
                prefix[:,:,:-2,:],         # prefix[n-1]
                M_changed[:,:,1:-1,:],     # изменённый слой
                suffix[:,:,2:,:]           # suffix[n+1]
            )
            # крайние слои
            M_total_changed[:,:,0,:] = np.einsum(
                'abw,bcw->acw',
                M_changed[:,:,0,:],     # (2,2,n_wavelengths)
                suffix[:,:,1,:]         # (2,2,n_wavelengths)
            )
            M_total_changed[:,:,-1,:] = np.einsum(
                'abw,bcw->acw',
                prefix[:,:,-2,:],       # (2,2,n_wavelengths)
                M_changed[:,:,-1,:]     # (2,2,n_wavelengths)
            )

            r, t = rt_amplitudes(M_total_changed, q_in, q_sub)
            R, T = RT_coeffs(r, t, q_in, q_sub)
            # print(rt_amplitudes(M_total_changed, q_in, q_sub))
            
            mf_new = rms_merit_layers(q_in, q_sub, wavelengths, targets, pol, theta_inc, r, t, R, T)
            best_idx = np.argmin(mf_new)

            if mf_new[best_idx] < mf:
                mf = mf_new[best_idx]
                thickness[best_idx] = thickness[best_idx] * (1.0 + sgn * step)
                phi[best_idx, :] = phi[best_idx, :] * (1.0 + sgn * step)
                improved = True
                break
        if not improved:
            step *= 0.5
            if step < min_step_rel:
                break

    res = make_stack(stack.start_flag, thickness, nH_values, nL_values,
                     cos_theta_in_H_layers, cos_theta_in_L_layers,
                     q_in, q_sub, qH, qL, wavelengths, n_wavelengths,
                     calculate_prefix_and_suffix_for_needle=True)

    return res, mf
