"""Verificación estructural (postprocesamiento). Unidades: MPa, mm.

Criterio MVP: tensión normal elástica máxima (N/A ± M·c/I) frente a la
fluencia del material de cada barra. FS = Sy / σ_max.
No incluye pandeo global, pandeo lateral-torsional, abolladura local,
corte combinado (von Mises) ni fatiga.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.model import Model
from core.solver import Results

FS_MIN_DEFAULT = 1.5


class Status(str, Enum):
    OK = "OK"
    ALERT = "ALERTA"  # 1 <= FS < FS_min
    FAIL = "FLUENCIA"  # FS < 1


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    fs: float  # factor de seguridad mínimo de la estructura
    fs_min: float  # FS admisible requerido
    sigma_max: float  # |σ| máxima [MPa]
    sy: float  # fluencia del material crítico [MPa]
    member: int  # barra crítica
    x: float  # posición local en la barra crítica [mm]
    point: tuple[float, float]  # posición global [mm]
    status: Status
    utilization: float  # σ_max / (Sy / FS_min), >1 no verifica

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


def check_safety(model: Model, results: Results, fs_min: float = FS_MIN_DEFAULT) -> SafetyCheck:
    """FS = Sy / σ_max evaluado barra por barra; devuelve la condición crítica."""
    best: SafetyCheck | None = None
    for mr in results.members:
        sy = model.members[mr.member].material.Sy
        sig = mr.sigma_abs
        k = int(np.argmax(sig))
        smax = float(sig[k])
        fs = sy / smax if smax > 1e-12 else float("inf")
        if best is None or fs < best.fs:
            status = Status.OK if fs >= fs_min else (Status.ALERT if fs >= 1.0 else Status.FAIL)
            best = SafetyCheck(
                fs=fs, fs_min=fs_min, sigma_max=smax, sy=sy, member=mr.member,
                x=float(mr.x[k]), point=(float(mr.points[k, 0]), float(mr.points[k, 1])),
                status=status, utilization=smax * fs_min / sy,
            )
    if best is None:
        raise ValueError("Resultados sin barras.")
    return best


def required_section_modulus(M_max: float, sy: float, fs: float = FS_MIN_DEFAULT) -> float:
    """W requerido [mm³] para |M_max| [N·mm] con FS dado (predimensionamiento a flexión pura)."""
    return abs(M_max) * fs / sy
