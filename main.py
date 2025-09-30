# main.py
import numpy as np
import time
from src.core.optics import n_of, n_cauchy, q_parameter, cos_theta_in_layer, dM_layer_dd_at_zero, RT_coeffs
from src.core.merit import rms_merit
from src.design.design import make_stack
from src.design.targets import target_AR, combine_targets, target_bandpass
from src.algorithms.needle import needle_cycle
from src.engine.report import print_report
import matplotlib.pyplot as plt

def plot_stack_spectra(stack, constants):
    wl_nm = constants["wavelengths"] * 1e9  # перевод в нм для удобства

    # fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # # Энергетические коэффициенты
    # axs.plot(wl_nm, stack.R, label="R (Reflectance)")
    # axs.plot(wl_nm, stack.T, label="T (Transmittance)")
    # axs.set_ylabel("R, T")
    # axs.legend()
    # axs.grid(True)

    # # Амплитудные коэффициенты (модуль и фаза)
    # axs[1].plot(wl_nm, np.abs(stack.r), label="|r|")
    # axs[1].plot(wl_nm, np.abs(stack.t), label="|t|")
    # axs[1].plot(wl_nm, np.angle(stack.r), "--", label="arg(r)")
    # axs[1].plot(wl_nm, np.angle(stack.t), "--", label="arg(t)")
    # axs[1].set_xlabel("Wavelength (nm)")
    # axs[1].set_ylabel("Amplitude / Phase")
    # axs[1].legend()
    # axs[1].grid(True)

    plt.figure(figsize=(8, 5))
    plt.plot(constants["wavelengths"] * 1e9, stack.R, label="R (Reflectance)")
    plt.plot(constants["wavelengths"] * 1e9, stack.T, label="T (Transmittance)")
    targets = constants["targets"]

    # если заданы цели
    if targets is not None:
        if "R" in targets:
            plt.plot(constants["wavelengths"] * 1e9,
                     targets["R"]["target"],
                     "r--", label="R target")
        if "T" in targets:
            plt.plot(constants["wavelengths"] * 1e9,
                     targets["T"]["target"],
                     "g--", label="target")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Coefficient")
    plt.title("Spectra of multilayer stack")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    constants = dict()

    constants["n_wavelengths"] = 1001
    constants["wavelengths"] = np.linspace(1000e-9, 1100e-9, constants["n_wavelengths"])
    quarter_at = 1050e-9
    constants["n_inc"] = np.array([n_of(n_cauchy, 1.0, wl) for wl in constants["wavelengths"]])
    constants["n_sub"] = np.array([n_of(n_cauchy, 1.52, wl) for wl in constants["wavelengths"]])
    constants["nH"] = np.array([n_of(n_cauchy, 2.35, wl) for wl in constants["wavelengths"]])
    constants["nL"] = np.array([n_of(n_cauchy, 1.45, wl) for wl in constants["wavelengths"]])
    dH = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 2.35, wl=1050e-9))))
    dL = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 1.45, wl=1050e-9))))
    constants["pol"] = "s"
    constants["theta_inc"]=0
    constants["cos_theta_in_inc"] = cos_theta_in_layer(constants["n_inc"], constants)
    constants["cos_theta_in_sub"] = cos_theta_in_layer(constants["n_sub"], constants)
    constants["cos_theta_in_H_layers"] = cos_theta_in_layer(constants["nH"], constants)
    constants["cos_theta_in_L_layers"] = cos_theta_in_layer(constants["nL"], constants)
    constants["q_in"] = q_parameter(constants["n_inc"], constants["cos_theta_in_inc"], constants)
    constants["q_sub"] = q_parameter(constants["n_sub"], constants["cos_theta_in_sub"], constants)
    constants["alpha"] = np.real(constants["q_sub"]/constants["q_in"])
    constants["qH"] = q_parameter(constants["nH"], constants["cos_theta_in_H_layers"], constants)
    constants["qL"] = q_parameter(constants["nL"], constants["cos_theta_in_L_layers"], constants)
    constants["kH"] = 2.0 * np.pi * constants["nH"] * constants["cos_theta_in_H_layers"] / constants["wavelengths"]
    constants["kL"] = 2.0 * np.pi * constants["nL"] * constants["cos_theta_in_L_layers"] / constants["wavelengths"]
    constants["dM_in_H_layer"] = dM_layer_dd_at_zero(constants["qH"], constants["kH"], constants)
    constants["dM_in_L_layer"] = dM_layer_dd_at_zero(constants["qL"], constants["kL"], constants)
    thickness = np.array([dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL])
    start_flag="H"

    # неизменны для данной задачи
    q_in = q_parameter(constants["n_inc"], np.cos(constants["theta_inc"]), constants)
    q_sub = q_parameter(constants["n_sub"], cos_theta_in_layer(constants["n_sub"], constants), constants)

    stack0 = make_stack(constants, start_flag, thickness, calculate_prefix_and_suffix_for_needle=True)

    constants["targets"] = combine_targets(target_bandpass(
        constants["wavelengths"],
        passbands=[(1040e-9, 1060e-9)],  # диапазон прозрачности
        sigma_pass=0.2,  # sigma в полосе
        sigma_stop=0.2   # sigma вне полосы
    ))
    
    old_merit = rms_merit(constants, stack0.r, stack0.t, stack0.R, stack0.T)
    print("old_MF")
    print(old_merit)

    constants["d_init"]=1e-9
    constants["d_eps"]=1e-10
    constants["step_rel"]=0.99
    constants["min_step_rel"]=0.01
    constants["iters"]=1000
    constants["d_min"]=0.5e-9
    constants["d_max"]=5e-7
    constants["max_steps"]=100
    constants["min_rel_improv"]=1e-4
    constants["max_layers"]=150
    constants["max_tot_nmopt"]=1e9
    constants["wl_ref_for_tot"]=1050e-9
    constants["verbose"]=True
    constants["log_timing"] = True
    t0 = time.perf_counter()
    stack, info = needle_cycle(constants, stack0)
    t1 = time.perf_counter()
    print("old_merit: "+str(old_merit))
    print("new_merit: "+str(rms_merit(constants, stack.r, stack.t, stack.R, stack.T)))
    print("info: "+str(info))
    print("stack:")
    print(stack.start_flag)
    for i in range(len(stack.thickness)):
        print(str(round(stack.thickness[i]/(1e-9), 3)))
    print("needle_cycle time: "+str(t1-t0))
    plot_stack_spectra(stack, constants)