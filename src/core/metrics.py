# src/core/metrics.py
import numpy as np
from ..core.optics import Stack

def total_optical_thickness(constants: dict, stack: Stack) -> float:
    """
    Считает суммарную оптическую толщину стека при опорной длине волны wl_ref_for_tot.
    Берёт показатель преломления из nH / nL при ближайшей длине волны.
    """
    # индекс ближайшей длины волны
    idx = int(np.argmin(np.abs(constants["wavelengths"] - constants["wl_ref_for_tot"])))
    nH = np.real(constants["nH"][idx])
    nL = np.real(constants["nL"][idx])

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
