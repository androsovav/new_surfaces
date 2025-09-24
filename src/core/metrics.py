# src/core/metrics.py
import numpy as np
from ..core.optics import Stack

def total_optical_thickness(
    stack: Stack,
    wl_ref: float,
    nH_values: np.ndarray,
    nL_values: np.ndarray,
    wavelengths: np.ndarray,
) -> float:
    """
    Считает суммарную оптическую толщину стека при опорной длине волны wl_ref.
    Берёт показатель преломления из nH_values / nL_values при ближайшей длине волны.
    """
    # индекс ближайшей длины волны
    idx = int(np.argmin(np.abs(wavelengths - wl_ref)))
    nH = np.real(nH_values[idx])
    nL = np.real(nL_values[idx])

    total = 0.0
    if stack.start_flag == "H":
        for i in range(len(stack.thickness)):
            n = nH if i % 2 == 0 else nL
            total += n * stack.thickness[i]
    else:
        for i in range(len(stack.thickness)):
            n = nL if i % 2 == 0 else nH
            total += n * stack.thickness[i]
    return float(total)
