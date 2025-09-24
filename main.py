# main.py
import numpy as np
import time
from src.core.optics import n_of, n_cauchy, q_parameter, cos_theta_in_layer, dM_layer_dd_at_zero
from src.core.merit import rms_merit
from src.design.design import make_stack
from src.design.targets import target_AR, combine_targets, target_bandpass
from src.algorithms.needle import needle_cycle
from src.algorithms.evolution import random_starts_search
from src.engine.report import print_report
import matplotlib.pyplot as plt

def plot_stack_spectra(stack, wavelengths):
    wl_nm = wavelengths * 1e9  # перевод в нм для удобства

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Энергетические коэффициенты
    axs[0].plot(wl_nm, stack.R, label="R (Reflectance)")
    axs[0].plot(wl_nm, stack.T, label="T (Transmittance)")
    axs[0].set_ylabel("R, T")
    axs[0].legend()
    axs[0].grid(True)

    # Амплитудные коэффициенты (модуль и фаза)
    axs[1].plot(wl_nm, np.abs(stack.r), label="|r|")
    axs[1].plot(wl_nm, np.abs(stack.t), label="|t|")
    axs[1].plot(wl_nm, np.angle(stack.r), "--", label="arg(r)")
    axs[1].plot(wl_nm, np.angle(stack.t), "--", label="arg(t)")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Amplitude / Phase")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_wavelengths = 1001
    wavelengths = np.linspace(1000e-9, 1100e-9, n_wavelengths)
    quarter_at = 1050e-9
    n_inc_values = np.array([n_of(n_cauchy, 1.0, wl) for wl in wavelengths])
    n_sub_values = np.array([n_of(n_cauchy, 1.52, wl) for wl in wavelengths])
    nH_values = np.array([n_of(n_cauchy, 2.35, wl) for wl in wavelengths])
    nL_values = np.array([n_of(n_cauchy, 1.45, wl) for wl in wavelengths])
    dH = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 2.35, wl=1050e-9))))
    dL = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 1.45, wl=1050e-9))))
    pol = "s"
    theta_inc=0
    cos_theta_in_inc = cos_theta_in_layer(n_inc_values, n_inc_values, theta_inc)
    cos_theta_in_sub = cos_theta_in_layer(n_sub_values, n_inc_values, theta_inc)
    cos_theta_in_H_layers = cos_theta_in_layer(nH_values, n_inc_values, theta_inc)
    cos_theta_in_L_layers = cos_theta_in_layer(nL_values, n_inc_values, theta_inc)
    q_in = q_parameter(n_inc_values, cos_theta_in_inc, pol)
    q_sub = q_parameter(n_sub_values, cos_theta_in_sub, pol)
    qH = q_parameter(nH_values, cos_theta_in_H_layers, pol)
    qL = q_parameter(nL_values, cos_theta_in_L_layers, pol)
    kH = 2.0 * np.pi * nH_values * cos_theta_in_H_layers / wavelengths
    kL = 2.0 * np.pi * nL_values * cos_theta_in_L_layers / wavelengths
    dM_in_H_layer = dM_layer_dd_at_zero(qH, kH, wavelengths)
    dM_in_L_layer = dM_layer_dd_at_zero(qL, kL, wavelengths)
    thickness = np.array([dH, dL, dH, dL, dH, dL, dH, dL])
    start_flag="H"

    # неизменны для данной задачи
    q_in = q_parameter(n_inc_values, np.cos(theta_inc), pol)
    q_sub = q_parameter(n_sub_values, cos_theta_in_layer(n_sub_values, n_inc_values, theta_inc), pol)
    stack0 = make_stack(start_flag=start_flag,
                    thickness = thickness,
                    nH_values=nH_values,
                    nL_values=nL_values,
                    cos_theta_in_H_layers=cos_theta_in_H_layers,
                    cos_theta_in_L_layers=cos_theta_in_L_layers,
                    q_in = q_in,
                    q_sub = q_sub,
                    qH=qH,
                    qL=qL,
                    wavelengths=wavelengths,
                    n_wavelengths=n_wavelengths,
                    calculate_prefix_and_suffix_for_needle=True)

    targets = targets = combine_targets(target_bandpass(
        wavelengths,
        passbands=[(1040e-9, 1060e-9)],  # диапазон прозрачности
        sigma_pass=0.2,  # sigma в полосе
        sigma_stop=0.2   # sigma вне полосы
    ))
    
    old_merit = rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, stack0.r, stack0.t, stack0.R, stack0.T)
    n_cands = [nH_values, nL_values]
    common_kwargs = dict(
        wavelengths=wavelengths,
        n_wavelengths=n_wavelengths,
        targets=targets,
        nH_values=nH_values,
        nL_values=nL_values,
        n_inc_values=n_inc_values,
        n_sub_values=n_sub_values,
        pol=pol,
        theta_inc=theta_inc,
        cos_theta_in_H_layers=cos_theta_in_H_layers,
        cos_theta_in_L_layers=cos_theta_in_L_layers,
        q_in=q_in,
        q_sub=q_sub,
        qH=qH,
        qL=qL,
        kH=kH,
        kL=kL,
        dM_in_H_layer=dM_in_H_layer,
        dM_in_L_layer=dM_in_L_layer,
        d_init=1e-10,
        d_eps=1e-11,
        coord_step_rel=3.0,
        coord_min_step_rel=1.01,
        coord_iters=1000,
        d_min=0.5e-9,
        max_steps=100,
        min_rel_improv=1e-4,
        max_layers=50,
        max_tot_nmopt=1e9,
        wl_ref_for_tot=1050e-9,
        verbose=True,
        log_timing = True
    )
    t0 = time.perf_counter()
    stack, info = needle_cycle(stack=stack0, **common_kwargs)
    t1 = time.perf_counter()
    print("old_merit: "+str(old_merit))
    print("new_merit: "+str(rms_merit(q_in, q_sub, wavelengths, targets, pol, theta_inc, stack.r, stack.t, stack.R, stack.T)))
    print("info: "+str(info))
    print("needle_cycle time: "+str(t1-t0))
    plot_stack_spectra(stack0, wavelengths)
    plot_stack_spectra(stack, wavelengths)