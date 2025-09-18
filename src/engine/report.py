# src/engine/report.py
from __future__ import annotations
from typing import Dict, Any, List
from ..core.metrics import layer_count, total_optical_thickness
from ..core.merit import rms_merit
from ..core.optics import Stack
import numpy as np

def summarize_result(
    stack: Stack,
    wavelengths: np.ndarray,
    targets: dict,
    pol: str,
    wl_ref: float,
    history: List[Dict[str, Any]] | None = None,
    elapsed: float | None = None,   # добавлено поле для времени
) -> str:
    """
    Формирует строковый отчёт о результате оптимизации.
    """
    mf = rms_merit(stack, wavelengths, targets, pol=pol)
    N = layer_count(stack)
    TOT = total_optical_thickness(stack, wl_ref) / 1e-9

    lines = []
    lines.append(f"MF={mf:.4f} | N={N} | TOT={TOT:.1f} nm-opt")
    if history:
        lines.append(f"History length: {len(history)} (MF first={history[0]['MF']:.4f}, last={history[-1]['MF']:.4f})")
    if elapsed is not None:
        lines.append(f"time: {elapsed:.3f} s")
    return "\n".join(lines)

def print_report(*args, **kwargs) -> None:
    print(summarize_result(*args, **kwargs))
