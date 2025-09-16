# main.py
import numpy as np
from design import make_stack
from targets import target_AR, target_bandpass, combine_targets
from needle import needle_cycle
from merit import rms_merit
from metrics import layer_count, total_optical_thickness

def demo():
    # спектральная сетка
    wl = np.linspace(450e-9, 650e-9, 301)

    # материалы
    n_air = 1.0
    n_sub = 1.52
    nH, nL = 2.35, 1.45

    # стартовый стек: 1 период HL
    stack = make_stack(n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL,
                       periods=1, quarter_at=550e-9)

    # --- цели ---
    # цель 1: минимальное отражение (AR)
    R_target = target_AR(wl, R_target=0.0, sigma=0.02)
    # цель 2: высокая прозрачность (T≈1)
    T_target = target_bandpass(
        wavelengths=wl,
        passbands=[(450e-9, 650e-9)],
        sigma_pass=0.02,
        sigma_stop=0.02
    )

    # комбинируем цели
    targets = combine_targets(R_target, T_target)

    # кандидаты материалов для иглы
    n_cands = [nH, nL]

    # запускаем needle-cycle
    result_stack, info = needle_cycle(
        stack=stack,
        wavelengths=wl,
        targets=targets,
        n_candidates=n_cands,
        pol="s",
        theta_inc=0.0,
        d_init=2e-9,
        d_eps=5e-10,
        coord_step_rel=0.25,
        coord_min_step_rel=0.01,
        coord_iters=60,
        d_min=0.5e-9,
        max_steps=20,
        min_rel_improv=1e-3,
        max_layers=60,
        max_tot_nmopt=8000.0,
        wl_ref_for_tot=550e-9,
    )

    # финальные метрики
    final_mf = rms_merit(result_stack, wl, targets, pol="s")
    N = layer_count(result_stack)
    TOT_nmopt = total_optical_thickness(result_stack, 550e-9) / 1e-9

    print("=== RESULT ===")
    print(f"Layers (N): {N}")
    print(f"TOT (nm-opt @550nm): {TOT_nmopt:.1f}")
    print(f"Final MF: {final_mf:.4f}")

    print("\n=== LOG (compact) ===")
    for rec in info["history"]:
        if rec["action"] == "init":
            print(f"step {rec['step']:02d}: init | MF={rec['MF']:.4f}, "
                  f"N={rec['N']}, TOT={rec['TOT_nmopt']:.1f}")
        elif rec["action"] == "needle+optimize":
            kind, idx = rec["pos"]
            print(f"step {rec['step']:02d}: add@{kind}[{idx}] n={rec['n_new']:.2f} "
                  f"dMF≈{rec['d_mf_pred']:.3e} -> MF {rec['MF_prev']:.4f} → {rec['MF']:.4f} "
                  f"(Δrel={rec['rel_improv']*100:.2f}%), N={rec['N']}, TOT={rec['TOT_nmopt']:.1f}")
        else:
            data = {k: rec[k] for k in rec if k not in ['step', 'action']}
            print(f"step {rec['step']:02d}: {rec['action']} | data={data}")

    return result_stack, info

if __name__ == "__main__":
    demo()
