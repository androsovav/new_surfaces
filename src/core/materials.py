# src/core/materials.py
from __future__ import annotations
from typing import Callable

# В первой версии — константные n(λ).
# Позже можно заменить дисперсией (Коши/Селльмейер или табличные данные).
def n_const(value: float) -> Callable[[float], complex]:
    return lambda wl: complex(value)
