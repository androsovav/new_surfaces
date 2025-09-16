# main.py
import numpy as np
from optics import Stack, Layer, RT
from design import make_stack
from materials import n_const
from targets import target_AR
from needle import needle_step
from optimizers import coordinate_descent_thicknesses
from merit import rms_merit
from metrics import layer_count, total_optical_thickness
from checks import energy_check

def demo():
    wl = np.linspace(450e-9, 650e-9, 301)  # метры
    n_air = 1.0
    n_sub = 1.52
    nH, nL = 2.35, 1.45

    # старт: никакой структуры (или 1 период HL)
    stack = make_stack(n_inc=n_air, n_sub=n_sub, nH=nH, nL=nL, periods=1, quarter_at=550e-9)

    wl_ref = 550e-9
    print("N:", layer_count(stack))
    print("TOT (nm-opt @550nm):", total_optical_thickness(stack, wl_ref)/1e-9)

    # Проверка энергии (для не-поглощающих n)
    wls = np.linspace(450e-9, 650e-9, 101)
    print("Max energy error (unpolarized):", energy_check(stack, wls, pol="u"))

    # цель: AR в диапазоне → минимизируем R(λ)
    target, sigma = target_AR(wl, R_target=0.0, sigma=0.02)

    print("Base layers:", len(stack.layers))
    base_mf = rms_merit(stack, wl, target, sigma, kind="R")
    print("Base MF:", base_mf)

    # Кандидаты материалов для иглы
    n_cands = [nH, nL]

    # 1) один игольный шаг
    stack, info = needle_step(stack, wl, target, sigma, n_candidates=n_cands, kind="R", d_init=2e-9, d_eps=0.5e-9)
    print("Inserted at pos", info["position"], "n=", info["n_new"], "pred dMF≈", info["d_mf_pred"])
    print("Layers after needle:", len(stack.layers))

    # 2) локальная доводка
    stack, mf = coordinate_descent_thicknesses(stack, wl, target, sigma, kind="R", step_rel=0.25, min_step_rel=0.01, iters=60, d_min=0.5e-9)
    print("After local optimize MF:", mf)

    # (Опционально) рассчитать финальный спектр
    R, T = RT(stack, wl)
    return wl, R, T, stack

if __name__ == "__main__":
    demo()
