"""Predimensionamiento: búsqueda del perfil más liviano que verifica.

Itera perfiles de una familia por área creciente, re-armando y resolviendo el
modelo completo (incluye el cambio de peso propio). El solver es exacto y
rápido (ms), así que la búsqueda exhaustiva es más simple y robusta que
estimar W o I requeridos. Un perfil se acepta sólo si verifica resistencia y servicio:

- Con `phi` (criterio de la app): verificación completa por combinaciones `check_combinations`,
  ELU σ_u ≤ φ·Sy por envolvente de 1.4D y 1.2D + 1.6L, y ELS flecha bajo 1.0D + 1.0L ≤ L/N.
- Sin `phi` (compatibilidad): `check_safety` con todas las cargas sin mayorar, FS ≥ FS mín. y L/N.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.materials import SECTIONS, Section
from core.model import Model
from core.solver import StructuralError, solve
from core.verification import (
    DEFLECTION_LIMIT_DEFAULT,
    CombinationCheck,
    SafetyCheck,
    check_combinations,
    check_safety,
    default_reference_length,
)


@dataclass(frozen=True, slots=True)
class Suggestion:
    section: Section
    check: SafetyCheck  # con combinaciones: verificación ELU de la combinación gobernante
    # ---- agregado al final (AGENTS.md §5) ---- #
    combination: CombinationCheck | None = None  # verificación completa ELU + ELS (si se usó `phi`)


def suggest_section(
    build: Callable[[Section], Model],
    family: str,
    fs_min: float,
    max_deflection: float | None = None,
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT,
    phi: float | None = None,
    *,
    stability: bool = False,
    unbraced_length: float | None = None,
    effective_length_factor_k: float = 1.0,
) -> Suggestion | None:
    """Perfil de menor área de `family` que verifica resistencia y flecha.

    Args:
        fs_min: FS mínimo admisible (resistencia). Se ignora si se da `phi`.
        max_deflection: flecha admisible absoluta [mm]. Si se da, reemplaza a
            `deflection_limit_ratio` (se convierte a N = L / max_deflection con la L del modelo).
        deflection_limit_ratio: N de la flecha admisible L/N (por defecto L/300); 0 = sin límite.
        phi: factor de resistencia φ ∈ (0; 1]. Si se da, cada candidato se verifica con
            `check_combinations` (ELU σ_u ≤ φ·Sy en 1.4D y 1.2D + 1.6L; ELS flecha con 1.0D + 1.0L)
            y la sugerencia trae `combination`. Es el mismo criterio que el veredicto de la app.

    Raises:
        ValueError: familia inexistente, flecha admisible ≤ 0 o φ fuera de (0; 1].
    """
    if family not in SECTIONS:
        raise ValueError(f"Familia '{family}' no válida en el catálogo.")
    if max_deflection is not None and max_deflection <= 0.0:
        raise ValueError("La flecha admisible debe ser positiva [mm].")
    if phi is not None and not 0.0 < phi <= 1.0:  # también rechaza NaN
        raise ValueError("El factor de resistencia φ debe estar en el intervalo (0; 1].")
    for sec in sorted(SECTIONS[family].values(), key=lambda s: s.A):
        model = build(sec)
        ratio = (default_reference_length(model) / max_deflection if max_deflection is not None
                 else deflection_limit_ratio)
        try:
            if phi is not None:
                comb = check_combinations(
                    model, phi=phi, deflection_limit_ratio=ratio, stability=stability,
                    unbraced_length=unbraced_length,
                    effective_length_factor_k=effective_length_factor_k,
                )
                if comb.ok:
                    return Suggestion(sec, comb.strength, comb)
                continue
            chk = check_safety(
                model, solve(model), fs_min, deflection_limit_ratio=ratio, stability=stability,
                unbraced_length=unbraced_length, effective_length_factor_k=effective_length_factor_k,
            )
        except StructuralError:
            return None
        if chk.ok:
            return Suggestion(sec, chk)
    return None
