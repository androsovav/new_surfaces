# main.py
import time
import numpy as np
from design import make_stack
from targets import target_AR, combine_targets
from merit import rms_merit
from metrics import layer_count, total_optical_thickness
from needle import needle_cycle

def demo_compare_speed():
    # Упрощённые параметры для быстрого теста
    wl = np.linspace(500e-9, 600e-9, 401)   # 81 точка вместо 301
    n_air = 1.0
    n_sub = 1.52
    nH, nL = 2.35, 1.45

    # старт: 1 период HL на 550 нм
    stack0 = make_stack(n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL, periods=1, quarter_at=550e-9)

    # цель: AR → минимизируем отражение (только R для простоты и скорости)
    targets = combine_targets(target_AR(wl, R_target=0.0, sigma=0.03))

    # кандидаты материалов для «иглы»
    n_cands = [nH, nL]

    common_kwargs = dict(
        wavelengths=wl,
        targets=targets,
        n_candidates=n_cands,
        pol="s",
        theta_inc=0.0,
        d_init=2e-9,
        d_eps=5e-10,
        coord_step_rel=0.25,
        coord_min_step_rel=0.02,
        coord_iters=0,         # <-- отключаем доводку
        d_min=0.5e-9,
        max_steps=3,           # <-- всего 3 шага
        min_rel_improv=1e-2,   # <-- агрессивный стоп
        max_layers=200,
        max_tot_nmopt=1e9,
        wl_ref_for_tot=550e-9,
        verbose=False,
    )

    # --- 1) Старый метод: дискретная P-карта ---
    stack_discr = make_stack(n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL, periods=10, quarter_at=550e-9)
    t0 = time.perf_counter()
    stack_discr, info_discr = needle_cycle(stack=stack_discr, pmap="discrete", **common_kwargs)
    t1 = time.perf_counter()

    mf_discr = rms_merit(stack_discr, wl, targets, pol="s")
    N_discr = layer_count(stack_discr)
    TOT_discr = total_optical_thickness(stack_discr, 550e-9) / 1e-9

    # --- 2) Новый метод: аналитическая P-карта ---
    stack_anal = make_stack(n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL, periods=10, quarter_at=550e-9)
    t2 = time.perf_counter()
    stack_anal, info_anal = needle_cycle(stack=stack_anal, pmap="analytic", **common_kwargs)
    t3 = time.perf_counter()

    mf_anal = rms_merit(stack_anal, wl, targets, pol="s")
    N_anal = layer_count(stack_anal)
    TOT_anal = total_optical_thickness(stack_anal, 550e-9) / 1e-9

    # печать результатов
    print("=== DISCRETE P-map ===")
    print(f"time: {t1 - t0:.3f} s | MF={mf_discr:.4f} | N={N_discr} | TOT={TOT_discr:.1f} nm-opt")
    print("=== ANALYTIC P-map ===")
    print(f"time: {t3 - t2:.3f} s | MF={mf_anal:.4f} | N={N_anal} | TOT={TOT_anal:.1f} nm-opt")

    if (t1 - t0) > 0:
        speedup = (t1 - t0) / max(t3 - t2, 1e-12)
        print(f"Speedup (analytic / discrete): ×{speedup:.2f}")

    return (stack_discr, info_discr), (stack_anal, info_anal)

if __name__ == "__main__":
    demo_compare_speed()
