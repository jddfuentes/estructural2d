"""Verificación estructural (postprocesamiento). Unidades: MPa, mm.

Criterios MVP:
- Resistencia: tensión normal elástica máxima (N/A ± M·c/I) frente a la fluencia
  del material de cada barra. FS = Sy / σ_max ≥ FS_min.
- Servicio (ELS): flecha relativa L/δ_max ≥ N (por defecto L/300), con δ_max el
  desplazamiento total máximo de la estructura y L la longitud de referencia
  (por defecto, el ancho horizontal del modelo = luz de la viga o del pórtico).

No incluye pandeo global, pandeo lateral-torsional, abolladura local,
corte combinado (von Mises) ni fatiga.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.model import Model
from core.solver import MemberResult, Results

FS_MIN_DEFAULT = 1.5

# Flecha admisible por defecto L/300: valor usual de predimensionamiento para vigas.
# Los límites de servicio dependen del uso y de la norma (AISC 360 cap. L y CIRSOC 301
# remiten a criterios de servicio por tipo de elemento): ajustar con `deflection_limit_ratio`.
DEFLECTION_LIMIT_DEFAULT = 300.0


class Status(str, Enum):
    OK = "OK"  # resistencia y servicio verifican
    ALERT = "ALERTA"  # 1 <= FS < FS_min, o flecha mayor a la admisible (sin fluencia)
    FAIL = "FLUENCIA"  # FS < 1 (domina sobre la flecha)


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    fs: float  # factor de seguridad mínimo de la estructura
    fs_min: float  # FS admisible requerido
    sigma_max: float  # |σ| máxima [MPa]
    sy: float  # fluencia del material crítico [MPa]
    member: int  # barra crítica
    x: float  # posición local en la barra crítica [mm]
    point: tuple[float, float]  # posición global [mm]
    status: Status  # veredicto global (resistencia + servicio)
    utilization: float  # σ_max / (Sy / FS_min), >1 no verifica
    # ---- Servicio (ELS) — campos agregados al final (AGENTS.md §5) ---- #
    delta_max: float = 0.0  # |δ| máxima de la estructura [mm]
    deflection_ratio: float = math.inf  # N = L / δ_max (flecha relativa L/N)
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT  # N admisible; 0 = sin límite
    deflection_ok: bool = True  # N >= N admisible  (δ_max <= L / N_adm)
    reference_length: float = 0.0  # L usada para L/δ [mm]

    @property
    def ok(self) -> bool:
        """Verifica resistencia y servicio."""
        return self.status is Status.OK

    @property
    def strength_ok(self) -> bool:
        """FS >= FS_min (sólo resistencia)."""
        return self.fs >= self.fs_min

    @property
    def delta_adm(self) -> float:
        """Flecha admisible δ_adm = L / N_adm [mm]; infinito si no hay límite."""
        if self.deflection_limit_ratio <= 0.0:
            return math.inf
        return self.reference_length / self.deflection_limit_ratio

    @property
    def issues(self) -> tuple[str, ...]:
        """Motivos por los que no es un OK limpio, en castellano (la UI los muestra tal cual)."""
        out: list[str] = []
        where = f"barra {self.member}, ({self.point[0] / 1e3:.2f}; {self.point[1] / 1e3:.2f}) m"
        if self.fs < 1.0:
            out.append(f"Fluencia: σ máx = {self.sigma_max:.1f} MPa supera Sy = {self.sy:g} MPa "
                       f"en {where}.")
        elif not self.strength_ok:
            out.append(f"Resistencia: FS = {self.fs:.2f} < FS mín. = {self.fs_min:g} "
                       f"(σ máx = {self.sigma_max:.1f} MPa en {where}).")
        if not self.deflection_ok:
            out.append(f"Servicio: flecha δ máx = {self.delta_max:.1f} mm supera δ adm = "
                       f"L/{self.deflection_limit_ratio:g} = {self.delta_adm:.1f} mm "
                       f"(L/δ = L/{self.deflection_ratio:,.0f}).")
        return tuple(out)


def default_reference_length(model: Model) -> float:
    """Longitud de referencia para L/δ [mm]: ancho horizontal del modelo (luz).

    Si el modelo no tiene extensión horizontal (p. ej. una columna aislada), usa la
    barra más larga.
    """
    xs = [x for x, _ in model.nodes]
    return max(xs) - min(xs) or max(model.member_length(m) for m in range(len(model.members)))


def check_safety(
    model: Model,
    results: Results,
    fs_min: float = FS_MIN_DEFAULT,
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT,
    reference_length: float | None = None,
) -> SafetyCheck:
    """Verificación de resistencia (FS = Sy / σ_max) y de servicio (L/δ_max ≥ N).

    Args:
        fs_min: factor de seguridad mínimo admisible.
        deflection_limit_ratio: N de la flecha admisible L/N (300 → L/300). 0 = sin límite de flecha.
        reference_length: L para L/δ [mm]; por defecto `default_reference_length(model)`.

    Veredicto: FLUENCIA si FS < 1; ALERTA si FS < fs_min o δ_max > L/N; OK si ambos verifican.

    Raises:
        ValueError: fs_min ≤ 0 (o NaN), N < 0 o L ≤ 0.
    """
    if not fs_min > 0.0:  # también rechaza NaN
        raise ValueError("El factor de seguridad admisible debe ser estrictamente positivo.")
    if deflection_limit_ratio < 0.0:
        raise ValueError("El límite de flecha L/N debe ser N ≥ 0 (0 = sin límite).")
    L_ref = default_reference_length(model) if reference_length is None else reference_length
    if L_ref <= 0.0:
        raise ValueError("La longitud de referencia para la flecha debe ser positiva [mm].")

    best: tuple[float, float, float, MemberResult, int] | None = None  # (fs, σ, Sy, barra, estación)
    for mr in results.members:
        sy = model.members[mr.member].material.Sy
        sig = mr.sigma_abs
        k = int(np.argmax(sig))
        smax = float(sig[k])
        fs = sy / smax if smax > 1e-12 else math.inf
        if best is None or fs < best[0]:
            best = (fs, smax, sy, mr, k)
    if best is None:
        raise ValueError("Resultados sin barras.")
    fs, smax, sy, mr, k = best

    delta = abs(results.extreme("deflection").value)
    ratio = L_ref / delta if delta > 1e-12 else math.inf
    deflection_ok = deflection_limit_ratio == 0.0 or ratio >= deflection_limit_ratio

    if fs < 1.0:
        status = Status.FAIL
    elif fs < fs_min or not deflection_ok:
        status = Status.ALERT
    else:
        status = Status.OK

    return SafetyCheck(
        fs=fs, fs_min=fs_min, sigma_max=smax, sy=sy, member=mr.member,
        x=float(mr.x[k]), point=(float(mr.points[k, 0]), float(mr.points[k, 1])),
        status=status, utilization=smax * fs_min / sy,
        delta_max=delta, deflection_ratio=ratio, deflection_limit_ratio=deflection_limit_ratio,
        deflection_ok=deflection_ok, reference_length=L_ref,
    )


def required_section_modulus(M_max: float, sy: float, fs: float = FS_MIN_DEFAULT) -> float:
    """W requerido [mm³] para |M_max| [N·mm] con FS dado (predimensionamiento a flexión pura)."""
    return abs(M_max) * fs / sy
