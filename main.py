# main.py
import numpy as np
import time
from src.core.optics import n_of, n_cauchy, q_parameter, cos_theta_in_layer, dM_layer_dd_at_zero, RT_coeffs
from src.core.merit import rms_merit
from src.design.design import make_stack
from src.design.targets import combine_targets, target_bandpass, target_ratio_RsRp
from src.algorithms.needle import needle_cycle
from src.engine.report import print_report
import matplotlib.pyplot as plt

def plot_stack_spectra(stack, constants, show_ratio: bool = True):
    """
    Рисует спектры R и T для каждой доступной поляризации (s/p),
    а также кривую отношения Rs/Rp и её целевую (если задана цель ratio_RsRp).
    Ожидается новая архитектура:
      stack.R = {"s": Rs(λ), "p": Rp(λ)}, аналогично T/r/t.
      constants["targets"] = {"s": {...}, "p": {...}, "ratio_RsRp": {...}}
    """
    wl_nm = constants["wavelengths"] * 1e9
    pols = [p for p in ("s", "p") if p in stack.R]

    # --- Фигура 1: R и T по поляризациям ---
    plt.figure(figsize=(9, 6))
    # R
    for pol in pols:
        plt.plot(wl_nm, stack.R[pol], label=f"R{pol}")
    # T
    for pol in pols:
        plt.plot(wl_nm, stack.T[pol], linestyle="--", label=f"T{pol}")

    # цели по поляризациям (если заданы)
    tg = constants.get("targets", {})
    for pol in pols:
        tp = tg.get(pol, {})
        if "R" in tp:
            plt.plot(wl_nm, tp["R"]["target"], linestyle=":", label=f"R{pol} target")
        if "T" in tp:
            plt.plot(wl_nm, tp["T"]["target"], linestyle=":", label=f"T{pol} target")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Coefficient")
    plt.title("Spectra (R, T) by polarization")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --- Фигура 2: отношение Rs/Rp (по желанию и если есть обе поляризации) ---
    if show_ratio and all(p in stack.R for p in ("s", "p")):
        plt.figure(figsize=(9, 4))
        ratio = (stack.R["s"]) / np.maximum(stack.R["p"], 1e-12)
        plt.plot(wl_nm, ratio, label="Rs/Rp")

        # цель по ratio, если задана
        if "ratio_RsRp" in tg:
            plt.plot(wl_nm, tg["ratio_RsRp"]["target"], linestyle=":", label="(Rs/Rp) target")

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Rs / Rp")
        plt.title("Polarization ratio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # === Общие константы ===
    constants = dict()
    constants["n_wavelengths"] = 1001
    constants["wavelengths"] = np.linspace(623e-9, 643e-9, constants["n_wavelengths"])
    quarter_at = 633e-9

    # Показатели преломления
    constants["n_inc"] = np.array([n_of(n_cauchy, 1.0, wl) for wl in constants["wavelengths"]])
    constants["n_sub"] = np.array([n_of(n_cauchy, 1.52, wl) for wl in constants["wavelengths"]])
    constants["nH"]    = np.array([n_of(n_cauchy, 2.35, wl) for wl in constants["wavelengths"]])
    constants["nL"]    = np.array([n_of(n_cauchy, 1.45, wl) for wl in constants["wavelengths"]])

    # Геометрия
    constants["theta_inc"] = 43 * np.pi / 180  # пример угла, можно изменить
    constants["cos_theta_in_inc"] = cos_theta_in_layer(constants["n_inc"], constants)
    constants["cos_theta_in_sub"] = cos_theta_in_layer(constants["n_sub"], constants)
    constants["cos_theta_in_H_layers"] = cos_theta_in_layer(constants["nH"], constants)
    constants["cos_theta_in_L_layers"] = cos_theta_in_layer(constants["nL"], constants)

    # Толщины четвертьволновые
    dH = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 2.35, wl=633e-9))))
    dL = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 1.45, wl=633e-9))))
    thickness = np.tile([dH, dL], 20)  # например, 40 слоев
    start_flag = "H"

    # === Поляризационно-зависимые параметры ===
    constants["q_in"]  = {}
    constants["q_sub"] = {}
    constants["alpha"] = {}
    constants["qH"] = {}
    constants["qL"] = {}
    constants["kH"] = {}
    constants["kL"] = {}
    constants["dM_in_H_layer"] = {}
    constants["dM_in_L_layer"] = {}

    for pol in ["s", "p"]:
        # локальная копия constants с выбранной поляризацией
        const_pol = {**constants, "pol": pol}

        # параметры входа и подложки
        q_in  = q_parameter(constants["n_inc"], constants["cos_theta_in_inc"], const_pol)
        q_sub = q_parameter(constants["n_sub"], constants["cos_theta_in_sub"], const_pol)

        constants["q_in"][pol] = q_in
        constants["q_sub"][pol] = q_sub
        constants["alpha"][pol] = np.real(q_sub / q_in)

        # параметры слоев
        qH = q_parameter(constants["nH"], constants["cos_theta_in_H_layers"], const_pol)
        qL = q_parameter(constants["nL"], constants["cos_theta_in_L_layers"], const_pol)

        constants["qH"][pol] = qH
        constants["qL"][pol] = qL

        constants["kH"][pol] = 2.0 * np.pi * constants["nH"] * constants["cos_theta_in_H_layers"] / constants["wavelengths"]
        constants["kL"][pol] = 2.0 * np.pi * constants["nL"] * constants["cos_theta_in_L_layers"] / constants["wavelengths"]

        constants["dM_in_H_layer"][pol] = dM_layer_dd_at_zero(qH, constants["kH"][pol], const_pol)
        constants["dM_in_L_layer"][pol] = dM_layer_dd_at_zero(qL, constants["kL"][pol], const_pol)
    thickness = np.array([dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL])
    start_flag="L"

    stack0 = make_stack(constants, start_flag, thickness, calculate_prefix_and_suffix_for_needle=True)

    wl0 = 633e-9
    half = 5e-9

    constants["targets"] = combine_targets(
        # Полное пропускание ~0.1% (T≈0.001) в полосе ±5 нм — для обеих поляризаций
        target_bandpass(
            wavelengths = constants["wavelengths"],
            passbands   = [(wl0 - half, wl0 + half)],
            T_in_pass   = 0.001,
            sigma_pass  = 2e-4,   # жёсткий штраф внутри полосы
            T_out       = 0.0,
            sigma_stop  = np.inf, # вне полосы без штрафа
            pols        = ("s","p")
        ),
        # Минимизируем Rs/Rp в той же полосе
        target_ratio_RsRp(
            wavelengths = constants["wavelengths"],
            bands       = [(wl0 - half, wl0 + half)],
            ratio_target= 0.0,    # «как можно меньше»
            sigma_in    = 0.05,   # подстрой под желаемый вес
            sigma_out   = np.inf
        )
    )

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
    constants["wl_ref_for_tot"]=633e-9
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
    plot_stack_spectra(stack, constants, show_ratio=True)