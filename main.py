# main.py
import numpy as np
import time
from src.design.design import make_stack
from src.design.targets import target_AR, combine_targets
from src.algorithms.needle import needle_cycle
from src.algorithms.evolution import random_starts_search
from src.engine.report import print_report


def run_needle_cycle():
    print("=== ANALYTIC P-map (single run) ===")
    wl = np.linspace(500e-9, 600e-9, 101)
    n_air = 1.0
    n_sub = 1.52
    nH, nL = 2.35, 1.45
    theta_inc=0.0
    pol="s"

    stack0 = make_stack(n_inc=n_air, 
                        n_sub=n_sub, 
                        nH=nH,
                        nL=nL,
                        theta_inc=theta_inc,
                        wl=wl,
                        pol=pol,
                        periods=1, 
                        quarter_at=550e-9)
    targets = combine_targets(target_AR(wl, R_target=0.0, sigma=0.03))
    n_cands = [nH, nL]

    common_kwargs = dict(
        wavelengths=wl,
        targets=targets,
        n_candidates=n_cands,
        pol=pol,
        theta_inc = theta_inc,
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
    stack, info = needle_cycle(stack=stack0, pmap="analytic", **common_kwargs)
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
    run_needle_cycle()
    run_random_search()