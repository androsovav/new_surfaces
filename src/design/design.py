# src/design/design.py
from __future__ import annotations
from typing import List, Callable, Literal
import numpy as np
import time
from ..core.optics import Stack, phi_parameter, make_M, rt_amplitudes, RT_coeffs

def make_stack(constants: dict,
               start_flag: Literal["H", "L"],
               thickness: np.ndarray,
               calculate_prefix_and_suffix_for_needle: bool) -> Stack:
    num_of_layers = len(thickness)
    K = constants["n_wavelengths"]

    # какие поляризации реально нужны (берём из targets, иначе — все доступные в qH)
    if "targets" in constants and any(k in constants["targets"] for k in ("s", "p")):
        pol_list = [k for k in ("s", "p") if k in constants["targets"]]
    else:
        pol_list = list(constants["qH"].keys())  # обычно ["s","p"]

    # фаза слоёв (общая для s/p)
    phi = np.empty((num_of_layers, K), dtype=np.complex128)
    if start_flag == "H":
        phi[0::2] = phi_parameter(constants["nH"], thickness[0::2],
                                  constants["cos_theta_in_H_layers"], constants["wavelengths"])
        phi[1::2] = phi_parameter(constants["nL"], thickness[1::2],
                                  constants["cos_theta_in_L_layers"], constants["wavelengths"])
    else:
        phi[0::2] = phi_parameter(constants["nL"], thickness[0::2],
                                  constants["cos_theta_in_L_layers"], constants["wavelengths"])
        phi[1::2] = phi_parameter(constants["nH"], thickness[1::2],
                                  constants["cos_theta_in_H_layers"], constants["wavelengths"])

    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # словари по поляризациям
    prefix, suffix, M, M_layers, q = {}, {}, {}, {}, {}

    for pol in pol_list:
        # q-раскладка по слоям с учётом стартовой буквы
        q_pol = np.empty((num_of_layers, K), dtype=np.complex128)
        if start_flag == "H":
            q_pol[0::2] = constants["qH"][pol]
            q_pol[1::2] = constants["qL"][pol]
        else:
            q_pol[0::2] = constants["qL"][pol]
            q_pol[1::2] = constants["qH"][pol]
        q[pol] = q_pol

        # матрицы слоёв
        M_layers_pol = make_M(sphi, cphi, q_pol, num_of_layers, K)
        M_layers[pol] = M_layers_pol

        if calculate_prefix_and_suffix_for_needle:
            # half-слои для prefix/suffix
            phi_half = 0.5 * phi
            sphi_half = np.sin(phi_half)
            cphi_half = np.cos(phi_half)
            M_half = make_M(sphi_half, cphi_half, q_pol, num_of_layers, K)

            prefix_pol = np.empty((2, 2, num_of_layers, K), dtype=complex)
            suffix_pol = np.empty((2, 2, num_of_layers, K), dtype=complex)

            left  = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))
            right = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))

            # prefix
            for i in range(num_of_layers):
                prefix_pol[:, :, i, :] = np.einsum('ijk,jlk->ilk', left, M_half[:, :, i, :])
                left = np.einsum('ijk,jlk->ilk', left, M_layers_pol[:, :, i, :])

            # suffix
            for i in range(num_of_layers - 1, -1, -1):
                suffix_pol[:, :, i, :] = np.einsum('ijk,jlk->ilk', M_half[:, :, i, :], right)
                right = np.einsum('ijk,jlk->ilk', M_layers_pol[:, :, i, :], right)

            prefix[pol] = prefix_pol
            suffix[pol] = suffix_pol
            M[pol] = left  # итоговая матрица стека
        else:
            # только итоговая матрица стека
            M_pol = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))
            for i in range(num_of_layers):
                M_pol = np.einsum('ijk,jlk->ilk', M_pol, M_layers_pol[:, :, i, :])
            M[pol] = M_pol
            prefix[pol] = []
            suffix[pol] = []

    # амплитуды и энергетические коэффициенты — функции должны возвращать словари по поляризациям
    rt = rt_amplitudes(constants, M)             # {"r": {"s":..., "p":...}, "t": {...}}
    RT = RT_coeffs(constants, rt["r"], rt["t"])  # {"R": {...}, "T": {...}}

    return Stack(
        thickness=thickness,
        start_flag=start_flag,
        prefix=prefix,
        suffix=suffix,
        M=M,
        phi=phi,
        sphi=sphi,
        cphi=cphi,
        M_layers=M_layers,
        r=rt["r"],
        t=rt["t"],
        R=RT["R"],
        T=RT["T"],
        q=q
    )

def add_prefix_and_suffix_to_stack(stack: Stack, n_wavelengths: int) -> Stack:
    """
    Дополняет существующий стек корректными prefix/suffix для КАЖДОЙ поляризации.
    Ожидается, что в stack уже есть:
      - phi (N, K) — общая для s/p
      - q[pol]           (N, K)
      - M_layers[pol]    (2,2,N,K)
    Возвращает новый Stack с заполненными prefix[pol], suffix[pol] формы (2,2,N,K).
    """
    N = len(stack.thickness)
    K = n_wavelengths

    phi_half = 0.5 * stack.phi
    sphi_half = np.sin(phi_half)
    cphi_half = np.cos(phi_half)

    prefix = {}
    suffix = {}

    for pol in stack.M_layers.keys():
        q_pol = stack.q[pol]                       # (N, K)
        M_layers_pol = stack.M_layers[pol]         # (2,2,N,K)

        # полу-слои для позиции вставки
        M_half = make_M(sphi_half, cphi_half, q_pol, N, K)   # (2,2,N,K)

        prefix_pol = np.empty((2, 2, N, K), dtype=complex)
        suffix_pol = np.empty((2, 2, N, K), dtype=complex)

        left  = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))   # (2,2,K)
        right = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, K))   # (2,2,K)

        # считаем prefix:  left · M_half(i); затем накапливаем left ·= M_layers(i)
        for i in range(N):
            prefix_pol[:, :, i, :] = np.einsum('ijk,jlk->ilk', left, M_half[:, :, i, :])
            left = np.einsum('ijk,jlk->ilk', left, M_layers_pol[:, :, i, :])

        # считаем suffix:  M_half(i) · right; затем накапливаем right = M_layers(i) · right (в обратном порядке)
        for i in range(N - 1, -1, -1):
            suffix_pol[:, :, i, :] = np.einsum('ijk,jlk->ilk', M_half[:, :, i, :], right)
            right = np.einsum('ijk,jlk->ilk', M_layers_pol[:, :, i, :], right)

        prefix[pol] = prefix_pol
        suffix[pol] = suffix_pol

    # Возвращаем новый Stack с обновлёнными prefix/suffix (остальное — как было)
    return Stack(
        thickness=stack.thickness,
        start_flag=stack.start_flag,
        prefix=prefix,
        suffix=suffix,
        M=stack.M,
        phi=stack.phi,
        sphi=stack.sphi,
        cphi=stack.cphi,
        M_layers=stack.M_layers,
        r=stack.r,
        t=stack.t,
        R=stack.R,
        T=stack.T,
        q=stack.q
    )