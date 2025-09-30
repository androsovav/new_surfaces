# src/design/design.py
from __future__ import annotations
from typing import List, Callable, Literal
import numpy as np
import time
from ..core.optics import Stack, phi_parameter, make_M, rt_amplitudes, RT_coeffs

def make_stack(constants: dict, start_flag: Literal["H", "L"], thickness: np.ndarray, calculate_prefix_and_suffix_for_needle: bool) -> Stack:
    num_of_layers = len(thickness)
    
    # Инициализация массивов для всех длин волн
    phi = np.empty((num_of_layers, constants["n_wavelengths"]), dtype=np.complex128)
    sphi = np.empty((num_of_layers, constants["n_wavelengths"]), dtype=np.complex128)
    cphi = np.empty((num_of_layers, constants["n_wavelengths"]), dtype=np.complex128)
    q = np.empty((num_of_layers, constants["n_wavelengths"]), dtype=np.complex128)

    if start_flag == "H":
        phi[0::2] = phi_parameter(constants["nH"], thickness[0::2], constants["cos_theta_in_H_layers"], constants["wavelengths"])
        phi[1::2] = phi_parameter(constants["nL"], thickness[1::2], constants["cos_theta_in_L_layers"], constants["wavelengths"])
        q[0::2] = constants["qH"]
        q[1::2] = constants["qL"]
    else:
        phi[0::2] = phi_parameter(constants["nL"], thickness[0::2], constants["cos_theta_in_L_layers"], constants["wavelengths"])
        phi[1::2] = phi_parameter(constants["nH"], thickness[1::2], constants["cos_theta_in_H_layers"], constants["wavelengths"])
        q[0::2] = constants["qL"]
        q[1::2] = constants["qH"]
    
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    
    M_layers = make_M(sphi, cphi, q, num_of_layers, constants["n_wavelengths"])

    if calculate_prefix_and_suffix_for_needle:
        prefix = np.empty((2,2,num_of_layers,constants["n_wavelengths"]), dtype=complex)
        suffix = np.empty((2,2,num_of_layers,constants["n_wavelengths"]), dtype=complex)
        phi_half = 0.5*phi
        sphi_half = np.sin(phi_half)
        cphi_half = np.cos(phi_half)
        M_half = make_M(sphi_half, cphi_half, q, num_of_layers, constants["n_wavelengths"])
        # единичная матрица для каждой длины волны
        

        # единичная матрица для каждой длины волны
        left  = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,constants["n_wavelengths"]))   # (2,2,n_wavelength)
        right = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,constants["n_wavelengths"]))   # (2,2,n_wavelength)

        # считаем prefix
        for i in range(num_of_layers):
            prefix[:,:,i,:] = np.einsum('ijk,jlk->ilk', left, M_half[:,:,i,:])
            left = np.einsum('ijk,jlk->ilk', left, M_layers[:,:,i,:])

        # считаем suffix
        for i in range(num_of_layers-1, -1, -1):
            suffix[:,:,i,:] = np.einsum('ijk,jlk->ilk', M_half[:,:,i,:], right)
            right = np.einsum('ijk,jlk->ilk', M_layers[:,:,i,:], right)
        
        M = left

    else:
        M = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,constants["n_wavelengths"]))
        for i in range(num_of_layers):
            M = np.einsum('ijk,jlk->ilk', M, M_layers[:,:,i,:])
        prefix = []
        suffix = []

    # амплитуды
    r, t = rt_amplitudes(constants, M)
    R, T = RT_coeffs(constants, r, t)
    
    return Stack(prefix=prefix, suffix=suffix, M=M, r=r, t=t, R=R, T=T, q=q,
                 phi=phi, sphi=sphi, cphi=cphi, M_layers=M_layers, thickness=thickness,
                 start_flag=start_flag)

def add_prefix_and_suffix_to_stack(stack: Stack, n_wavelengths: int):
    num_of_layers = len(stack.thickness)
    prefix = np.empty((2,2,num_of_layers,n_wavelengths), dtype=complex)
    suffix = np.empty((2,2,num_of_layers,n_wavelengths), dtype=complex)
    phi_half = 0.5*stack.phi
    sphi_half = np.sin(phi_half)
    cphi_half = np.cos(phi_half)
    M_layers = stack.M_layers
    M_half = make_M(sphi_half, cphi_half, stack.q, num_of_layers, n_wavelengths)
    # единичная матрица для каждой длины волны
    

    # единичная матрица для каждой длины волны
    left  = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,n_wavelengths))   # (2,2,n_wavelength)
    right = np.tile(np.eye(2, dtype=complex)[:,:,None], (1,1,n_wavelengths))   # (2,2,n_wavelength)

    # считаем prefix
    for i in range(num_of_layers):
        prefix[:,:,i,:] = np.einsum('ijk,jlk->ilk', left, M_half[:,:,i,:])
        left = np.einsum('ijk,jlk->ilk', left, M_layers[:,:,i,:])

    # считаем suffix
    for i in range(num_of_layers-1, -1, -1):
        suffix[:,:,i,:] = np.einsum('ijk,jlk->ilk', M_half[:,:,i,:], right)
        right = np.einsum('ijk,jlk->ilk', M_layers[:,:,i,:], right)
    
    return Stack(prefix=prefix, suffix=suffix, M=stack.M, r=stack.r, t=stack.t, R=stack.R, T=stack.T,
                 phi=stack.phi, sphi=stack.sphi, cphi=stack.cphi, M_layers=M_layers, thickness=stack.thickness,
                 start_flag=stack.start_flag, q = stack.q)