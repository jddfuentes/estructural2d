"""Predimensionamiento: búsqueda del perfil más liviano que verifica.

Itera perfiles de una familia por área creciente, re-armando y resolviendo el
modelo completo (incluye el cambio de peso propio). El solver es exacto y
rápido (ms), así que la búsqueda exhaustiva es más simple y robusta que
estimar W o I requeridos. Un perfil se acepta sólo si verifica resistencia
(FS ≥ FS mín.) y servicio (flecha L/δ ≥ N), con el mismo `check_safety` que la app.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.materials import SECTIONS, Section
from core.model import Model
from core.solver import StructuralError, solve
from core.verification import DEFLECTION_LIMIT_DEFAULT, SafetyCheck, check_safety, default_reference_length


@dataclass(frozen=True, slots=True)
class Suggestion:
    section: Section
    check: SafetyCheck


def suggest_section(
    build: Callable[[Section], Model],
    family: str,
    fs_min: float,
    max_deflection: float | None = None,
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT,
) -> Suggestion | None:
    """Perfil de menor área de `family` que verifica resistencia y flecha.

    Args:
        fs_min: FS mínimo admisible (resistencia).
        max_deflection: flecha admisible absoluta [mm]. Si se da, reemplaza a
            `deflection_limit_ratio` (se convierte a N = L / max_deflection con la L del modelo).
        deflection_limit_ratio: N de la flecha admisible L/N (por defecto L/300); 0 = sin límite.
    """
    if max_deflection is not None and max_deflection <= 0.0:
        raise ValueError("La flecha admisible debe ser positiva [mm].")
    for sec in sorted(SECTIONS[family].values(), key=lambda s: s.A):
        model = build(sec)
        try:
            res = solve(model)
        except StructuralError:
            return None
        ratio = (default_reference_length(model) / max_deflection if max_deflection is not None
                 else deflection_limit_ratio)
        chk = check_safety(model, res, fs_min, deflection_limit_ratio=ratio)
        if chk.ok:
            return Suggestion(sec, chk)
    return None
