# needle.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal, Dict, Any
from optics import Stack
from design import insert_layer, insert_with_split
from merit import rms_merit
from optimizers import coordinate_descent_thicknesses
from metrics import layer_count, total_optical_thickness

PositionKind = Literal["interface", "split"]

def _test_insert(stack: Stack, pos: Tuple[PositionKind, int], n_new: float, d_new: float) -> Stack:
    kind, idx = pos
    if kind == "interface":
        return insert_layer(stack, index=idx, n_new=n_new, d_new=d_new)
    else:
        return insert_with_split(stack, layer_index=idx, n_new=n_new, d_new=d_new, split_ratio=0.5)

def enumerate_positions(stack: Stack) -> List[Tuple[PositionKind, int]]:
    N = len(stack.layers)
    positions: List[Tuple[PositionKind, int]] = []
    positions += [("interface", i) for i in range(N + 1)]
    positions += [("split", i) for i in range(N)]
    return positions

def discrete_excitation_map(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: List[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    d_eps: float = 5e-10,
    d_min: float | None = None,
) -> Tuple[List[Tuple[PositionKind,int]], np.ndarray, np.ndarray]:
    base_mf = rms_merit(stack, wavelengths, targets, pol=pol, theta_inc=theta_inc)
    positions = enumerate_positions(stack)
    n_best = np.full(len(positions), np.nan, dtype=float)
    dmf_best = np.full(len(positions), np.inf, dtype=float)

    for i, pos in enumerate(positions):
        for n_new in n_candidates:
            if d_min is not None and d_eps < d_min:
                continue
            test = _test_insert(stack, pos, n_new=n_new, d_new=d_eps)
            mf = rms_merit(test, wavelengths, targets, pol=pol, theta_inc=theta_inc)
            dmf = mf - base_mf
            if dmf < dmf_best[i]:
                dmf_best[i] = dmf
                n_best[i] = n_new
    return positions, n_best, dmf_best

def needle_cycle(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    n_candidates: List[float],
    pol: str = "s",
    theta_inc: float = 0.0,
    d_init: float = 2e-9,
    d_eps: float = 5e-10,
    coord_step_rel: float = 0.25,
    coord_min_step_rel: float = 0.01,
    coord_iters: int = 60,
    d_min: float | None = 0.5e-9,
    d_max: float | None = None,
    max_steps: int = 30,
    min_rel_improv: float = 1e-3,
    max_layers: int | None = None,
    max_tot_nmopt: float | None = None,
    wl_ref_for_tot: float = 550e-9,
) -> Tuple[Stack, Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    mf = rms_merit(stack, wavelengths, targets, pol=pol, theta_inc=theta_inc)
    N = layer_count(stack)
    TOT_nmopt = total_optical_thickness(stack, wl_ref_for_tot) / 1e-9
    history.append({"step": 0, "action": "init", "MF": mf, "N": N, "TOT_nmopt": TOT_nmopt})
    print(f"step 00: init | MF={mf:.4f}, N={N}, TOT={TOT_nmopt:.1f}")

    for step in range(1, max_steps + 1):
        positions, n_best, dmf_best = discrete_excitation_map(
            stack, wavelengths, targets,
            n_candidates=n_candidates, pol=pol, theta_inc=theta_inc,
            d_eps=d_eps, d_min=d_min
        )
        j = int(np.argmin(dmf_best))
        if not np.isfinite(dmf_best[j]) or dmf_best[j] >= 0.0:
            history.append({"step": step, "action": "stop_no_negative_P", "best_dMF": float(dmf_best[j])})
            print(f"step {step:02d}: stop_no_negative_P | best_dMF={dmf_best[j]:.3e}")
            break

        pos = positions[j]
        d_insert = max(d_init, (d_min if d_min is not None else d_init))
        stack = _test_insert(stack, pos, n_new=float(n_best[j]), d_new=d_insert)

        stack, mf_new = coordinate_descent_thicknesses(
            stack, wavelengths, targets, pol=pol, theta_inc=theta_inc,
            step_rel=coord_step_rel, min_step_rel=coord_min_step_rel, iters=coord_iters,
            d_min=d_min, d_max=d_max
        )

        N = layer_count(stack)
        TOT_nmopt = total_optical_thickness(stack, wl_ref_for_tot) / 1e-9
        rel_improv = (mf - mf_new) / max(mf, 1e-12)
        history.append({
            "step": step, "action": "needle+optimize",
            "pos": pos, "n_new": float(n_best[j]), "d_mf_pred": float(dmf_best[j]),
            "MF": float(mf_new), "MF_prev": float(mf), "rel_improv": float(rel_improv),
            "N": int(N), "TOT_nmopt": float(TOT_nmopt),
        })

        kind, idx = pos
        print(f"step {step:02d}: add@{kind}[{idx}] n={n_best[j]:.2f} "
              f"dMF≈{dmf_best[j]:.3e} -> MF {mf:.4f} → {mf_new:.4f} "
              f"(Δrel={rel_improv*100:.2f}%), N={N}, TOT={TOT_nmopt:.1f}")

        mf = mf_new

        if rel_improv < min_rel_improv:
            history.append({"step": step, "action": "stop_small_improv", "rel_improv": float(rel_improv)})
            print(f"step {step:02d}: stop_small_improv | rel_improv={rel_improv:.3e}")
            break
        if max_layers is not None and N > max_layers:
            history.append({"step": step, "action": "stop_layers_limit", "N": int(N), "max_layers": int(max_layers)})
            print(f"step {step:02d}: stop_layers_limit | N={N} > {max_layers}")
            break
        if max_tot_nmopt is not None and TOT_nmopt > max_tot_nmopt:
            history.append({"step": step, "action": "stop_tot_limit", "TOT_nmopt": float(TOT_nmopt), "max_tot_nmopt": float(max_tot_nmopt)})
            print(f"step {step:02d}: stop_tot_limit | TOT={TOT_nmopt:.1f} > {max_tot_nmopt}")
            break

    return stack, {"history": history}
