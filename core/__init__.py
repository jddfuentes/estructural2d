"""Núcleo de cálculo estructural 2D (sin dependencias de UI).

Convención de unidades en TODO el core: mm, N, MPa (N/mm²), N·mm.
"""

from core.materials import MATERIALS, SECTIONS, Material, Section
from core.model import (
    DistributedLoad,
    LoadCase,
    LoadCombination,
    LoadDirection,
    Member,
    Model,
    NodalLoad,
    Spring,
    Support,
    SupportType,
)
from core.solver import Results, StructuralError, combine_results, solve, solve_combinations, solve_load_cases
from core.verification import CombinationCheck, SafetyCheck, check_combinations, check_safety

__all__ = [
    "MATERIALS",
    "SECTIONS",
    "CombinationCheck",
    "DistributedLoad",
    "LoadCase",
    "LoadCombination",
    "LoadDirection",
    "Material",
    "Member",
    "Model",
    "NodalLoad",
    "Results",
    "SafetyCheck",
    "Section",
    "Spring",
    "StructuralError",
    "Support",
    "SupportType",
    "check_combinations",
    "check_safety",
    "combine_results",
    "solve",
    "solve_combinations",
    "solve_load_cases",
]
