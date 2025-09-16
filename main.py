# main.py
import numpy as np
from design import make_stack
from targets import target_AR, target_bandpass, combine_targets
from merit import rms_merit
from metrics import layer_count, total_optical_thickness
from evolution import random_starts_search

def demo():
    wl = np.linspace(450e-9, 650e-9, 301)

    # материалы
    n_air = 1.0
    n_sub = 1.52
    nH, nL = 2.35, 1.45

    # цель: R минимально + T максимально в диапазоне
    R_target = target_AR(wl, R_target=0.0, sigma=0.02)
    T_target = target_bandpass(wl, [(450e-9, 650e-9)], sigma_pass=0.02, sigma_stop=0.02)
    targets = combine_targets(R_target, T_target)

    # кандидаты для иглы
    n_cands = [nH, nL]

    # параметры ограничений и угла/поляризации
    theta_deg = 30.0          # угол падения, градусы
    theta_inc = np.deg2rad(theta_deg)
    pol = "u"                 # неполяризованный (усреднение s/p)

    # случайные старты + последовательная эволюция
    best_stack, info = random_starts_search(
        starts=5,
        n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL,
        wl0=550e-9,
        N_layers_range=(4, 10),
        d_min=0.5e-9, d_max=200e-9,
        wavelengths=wl,
        targets=targets,
        n_candidates=n_cands,
        pol=pol, theta_inc=theta_inc,
        evolution="sequential",
        evolution_kwargs={"steps": 3, "step_growth": 1.05, "wl_ref_for_tot": 550e-9},
        needle_kwargs={
            "d_init": 2e-9, "d_eps": 5e-10,
            "coord_step_rel": 0.25, "coord_min_step_rel": 0.01, "coord_iters": 60,
            "max_steps": 15, "min_rel_improv": 1e-3,
            "max_layers": 80, "max_tot_nmopt": 12000.0, "wl_ref_for_tot": 550e-9,
        },
        rng=np.random.default_rng(42),
    )

    final_mf = rms_merit(best_stack, wl, targets, pol=pol, theta_inc=theta_inc)
    N = layer_count(best_stack)
    TOT_nmopt = total_optical_thickness(best_stack, 550e-9) / 1e-9

    print("=== RESULT ===")
    print(f"Incidence angle: {theta_deg:.1f} deg, pol={pol}")
    print(f"Layers (N): {N}")
    print(f"TOT (nm-opt @550nm): {TOT_nmopt:.1f}")
    print(f"Final MF: {final_mf:.4f}")

    print("\n=== LOG (compact) ===")
    for rec in info["history"]:
        if rec["action"] == "init":
            data = {"MF": round(rec["MF"], 4), "N": rec["N"], "TOT_nmopt": round(rec["TOT_nmopt"], 1)}
            print(f"step {rec['step']:02d}: init | data={data}")
        elif rec["action"] == "needle+optimize":
            kind, idx = rec["pos"]
            data = dict(MF_prev=round(rec["MF_prev"],4), MF=round(rec["MF"],4),
                        rel_improv=round(rec["rel_improv"]*100,2),
                        n_new=round(rec["n_new"],2),
                        d_mf_pred=rec["d_mf_pred"],
                        N=rec["N"], TOT_nmopt=round(rec["TOT_nmopt"],1))
            print(f"step {rec['step']:02d}: add@{kind}[{idx}] | data={data}")
        else:
            data = {k: rec[k] for k in rec if k not in ["step","action"]}
            print(f"step {rec['step']:02d}: {rec['action']} | data={data}")

    return best_stack, info

if __name__ == "__main__":
    demo()
