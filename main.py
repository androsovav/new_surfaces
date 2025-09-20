# main.py
import numpy as np
import time
from src.core.optics import n_of, n_cauchy, q_parameter, cos_theta_in_layer
from src.design.design import make_stack
from src.design.targets import target_AR, combine_targets
from src.algorithms.needle import needle_cycle
from src.algorithms.evolution import random_starts_search
from src.engine.report import print_report


def run_needle_cycle():
    print("=== ANALYTIC P-map (single run) ===")
    wavelengths = np.linspace(500e-9, 600e-9, 101)
    quarter_at = 550e-9
    n_inc = 1.0
    n_sub = 1.52
    nH = 2.35
    nL = 1.45
    dH = (quarter_at / (4.0 * n_of(nspec=nH, wl=550e-9)))
    dL = (quarter_at / (4.0 * n_of(nspec=nL, wl=550e-9)))
    pol = "s"
    theta_inc=0.5
    q_in = q_parameter(n_inc, np.cos(theta_inc), pol)
    q_sub = q_parameter(n_sub, cos_theta_in_layer(n_sub, n_inc, theta_inc), pol)
    nH_values = np.array([n_of(nH, wl) for wl in wavelengths])
    nL_values = np.array([n_of(nH, wl) for wl in wavelengths])
    stack0 = make_stack(start_flag="H",
                    thickness = np.array([dH, dL]),
                    n_inc=n_inc, 
                    n_sub=n_sub, 
                    nH_values=nH_values,
                    nL_values=nL_values,
                    theta_inc=theta_inc,
                    wavelengths=wavelengths,
                    pol=pol)
    targets = combine_targets(target_AR(wl, R_target=0.0, sigma=0.03))
    n_cands = [nH, nL]

    common_kwargs = dict(
        wavelengths=wl,
        targets=targets,
        n_candidates=n_cands,
        pol=pol,
        theta_inc = theta_inc,
        q_in=q_in,
        q_sub=q_sub,
        d_init=2e-9,
        d_eps=5e-10,
        coord_step_rel=0.25,
        coord_min_step_rel=0.02,
        coord_iters=3,
        d_min=0.5e-9,
        max_steps=5,
        min_rel_improv=1e-2,
        max_layers=200,
        max_tot_nmopt=1e9,
        wl_ref_for_tot=550e-9,
        verbose=True,
    )

    t0 = time.perf_counter()
    stack, info = needle_cycle(stack=stack0, **common_kwargs)
    elapsed = time.perf_counter() - t0

    print_report(stack, wl, targets, pol="s", wl_ref=550e-9,
                 history=info.get("history"), elapsed=elapsed)


def run_random_search():
    print("\n=== RANDOM STARTS SEARCH (with evolution) ===")
    wl = np.linspace(500e-9, 600e-9, 101)
    n_air = 1.0
    n_sub = 1.52
    nH, nL = 2.35, 1.45

    targets = combine_targets(target_AR(wl, R_target=0.0, sigma=0.03))
    n_cands = [nH, nL]

    t0 = time.perf_counter()
    stack, info = random_starts_search(
        starts=5,  # число случайных стартов
        n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL,
        wl0=550e-9, N_layers_range=(4, 8),
        d_min=0.5e-9, d_max=50e-9,
        wavelengths=wl, targets=targets, n_candidates=n_cands,
        pol="s", theta_inc=0.0,
        evolution="sequential",  # sequential | gradual | none
        evolution_kwargs=dict(steps=2, step_growth=1.05),
        needle_kwargs=dict(coord_iters=3),
    )
    elapsed = time.perf_counter() - t0

    print_report(stack, wl, targets, pol="s", wl_ref=550e-9,
                 history=info.get("history"), elapsed=elapsed)


if __name__ == "__main__":
    n_wavelengths = 2
    wavelengths = np.linspace(500e-9, 600e-9, n_wavelengths)
    quarter_at = 550e-9
    n_inc_values = np.array([n_of(n_cauchy, 1.0, wl) for wl in wavelengths])
    n_sub_values = np.array([n_of(n_cauchy, 1.52, wl) for wl in wavelengths])
    nH_values = np.array([n_of(n_cauchy, 2.35, wl) for wl in wavelengths])
    nL_values = np.array([n_of(n_cauchy, 1.45, wl) for wl in wavelengths])
    dH = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 2.35, wl=550e-9))))
    dL = (quarter_at / (4.0 * np.real(n_of(n_cauchy, 1.45, wl=550e-9))))
    pol = "s"
    theta_inc=0.5
    cos_theta_in_H_layers = cos_theta_in_layer(nH_values, n_inc_values, theta_inc)
    cos_theta_in_L_layers = cos_theta_in_layer(nL_values, n_inc_values, theta_inc)
    qH = q_parameter(nH_values, cos_theta_in_H_layers, pol)
    qL = q_parameter(nL_values, cos_theta_in_L_layers, pol)

    # неизменны для данной задачи
    q_in = q_parameter(n_inc_values, np.cos(theta_inc), pol)
    q_sub = q_parameter(n_inc_values, cos_theta_in_layer(n_sub_values, n_inc_values, theta_inc), pol)
    t0 = time.perf_counter()
    stack0 = make_stack(start_flag="H",
                    thickness = np.array([dH, dL]),
                    n_inc_values=n_inc_values, 
                    n_sub_values=n_sub_values, 
                    nH_values=nH_values,
                    nL_values=nL_values,
                    cos_theta_in_H_layers=cos_theta_in_H_layers,
                    cos_theta_in_L_layers=cos_theta_in_L_layers,
                    qH=qH,
                    qL=qL,
                    wavelengths=wavelengths,
                    n_wavelengths=n_wavelengths,
                    pol=pol)
    print("time")
    t1 = time.perf_counter()
    print(t1-t0)
    print("stack0.M")
    print(stack0.M)
    print("prefix*suffix")
    print(stack0.prefix[0]*stack0.suffix[0])