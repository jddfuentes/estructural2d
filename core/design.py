"""Predimensionamiento: búsqueda del perfil más liviano que verifica.

Itera perfiles de una familia por área creciente, re-armando y resolviendo el
modelo completo (incluye el cambio de peso propio). El solver es exacto y
rápido (ms), así que la búsqueda exhaustiva es más simple y robusta que
estimar W requerido.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.materials import SECTIONS, Section
from core.model import Model
from core.solver import StructuralError, solve
from core.verification import SafetyCheck, check_safety


@dataclass(frozen=True, slots=True)
class Suggestion:
    section: Section
    check: SafetyCheck


def suggest_section(
    build: Callable[[Section], Model],
    family: str,
    fs_min: float,
    max_deflection: float | None = None,
) -> Suggestion | None:
    """Perfil de menor área de `family` con FS >= fs_min (y |δ| <= max_deflection [mm] si se da)."""
    for sec in sorted(SECTIONS[family].values(), key=lambda s: s.A):
        model = build(sec)
        try:
            res = solve(model)
        except StructuralError:
            return None
        chk = check_safety(model, res, fs_min)
        if not chk.ok:
            continue
        if max_deflection is not None and abs(res.extreme("deflection").value) > max_deflection:
            continue
        return Suggestion(sec, chk)
    return None
