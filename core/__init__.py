"""Núcleo de cálculo estructural 2D (sin dependencias de UI).

Convención de unidades en TODO el core: mm, N, MPa (N/mm²), N·mm.
"""

from core.materials import MATERIALS, SECTIONS, Material, Section
from core.model import (
    DistributedLoad,
    LoadDirection,
    Member,
    Model,
    NodalLoad,
    Support,
    SupportType,
)
from core.solver import Results, StructuralError, solve
from core.verification import SafetyCheck, check_safety

__all__ = [
    "MATERIALS",
    "SECTIONS",
    "DistributedLoad",
    "LoadDirection",
    "Material",
    "Member",
    "Model",
    "NodalLoad",
    "Results",
    "SafetyCheck",
    "Section",
    "StructuralError",
    "Support",
    "SupportType",
    "check_safety",
    "solve",
]
