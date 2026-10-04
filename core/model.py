"""Modelo de datos de la estructura 2D (pórtico plano).

Sistema global: X horizontal a la derecha, Y vertical hacia arriba, giro Z
antihorario positivo. Unidades: mm, N, N/mm, N·mm.

Cada nodo tiene 3 GDL: (ux, uy, rz). Las barras son elementos de pórtico
Euler-Bernoulli con rigidez axial y a flexión.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from core.materials import Material, Section


class SupportType(str, Enum):
    FIXED = "empotrado"  # restringe ux, uy, rz
    PINNED = "articulado"  # restringe ux, uy
    ROLLER = "móvil"  # restringe uy (rueda sobre superficie horizontal)

    @property
    def restrained_dofs(self) -> tuple[bool, bool, bool]:
        return {
            SupportType.FIXED: (True, True, True),
            SupportType.PINNED: (True, True, False),
            SupportType.ROLLER: (False, True, False),
        }[self]


class LoadDirection(str, Enum):
    LOCAL_Y = "local_y"  # perpendicular a la barra (y local)
    GLOBAL_Y = "global_y"  # vertical global, por unidad de longitud de barra
    GLOBAL_X = "global_x"  # horizontal global, por unidad de longitud de barra


@dataclass(frozen=True, slots=True)
class Member:
    i: int  # nodo inicial
    j: int  # nodo final
    section: Section
    material: Material


@dataclass(frozen=True, slots=True)
class Support:
    node: int
    type: SupportType


@dataclass(frozen=True, slots=True)
class NodalLoad:
    """Carga concentrada en nodo, ejes globales. Fy < 0 = hacia abajo."""

    node: int
    Fx: float = 0.0  # [N]
    Fy: float = 0.0  # [N]
    Mz: float = 0.0  # [N·mm], antihorario +


@dataclass(frozen=True, slots=True)
class DistributedLoad:
    """Carga distribuida lineal (trapezoidal) sobre toda la barra.

    q_start / q_end [N/mm] (= kN/m) en el nodo i y j de la barra.
    Signo según `direction`: con GLOBAL_Y, negativo = hacia abajo.
    """

    member: int
    q_start: float
    q_end: float
    direction: LoadDirection = LoadDirection.GLOBAL_Y
    label: str = ""  # etiqueta informativa (p.ej. SELF_WEIGHT); no afecta el cálculo


SELF_WEIGHT = "peso propio"


@dataclass(slots=True)
class Model:
    nodes: list[tuple[float, float]] = field(default_factory=list)
    members: list[Member] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    nodal_loads: list[NodalLoad] = field(default_factory=list)
    distributed_loads: list[DistributedLoad] = field(default_factory=list)

    @property
    def n_dof(self) -> int:
        return 3 * len(self.nodes)

    def member_length(self, m: int) -> float:
        mem = self.members[m]
        (x1, y1), (x2, y2) = self.nodes[mem.i], self.nodes[mem.j]
        return math.hypot(x2 - x1, y2 - y1)

    def member_cos_sin(self, m: int) -> tuple[float, float]:
        mem = self.members[m]
        (x1, y1), (x2, y2) = self.nodes[mem.i], self.nodes[mem.j]
        L = math.hypot(x2 - x1, y2 - y1)
        return (x2 - x1) / L, (y2 - y1) / L

    def is_horizontal_beam(self) -> bool:
        """True si todos los nodos están sobre y = cte (viga continua)."""
        ys = {round(y, 9) for _, y in self.nodes}
        return len(ys) == 1

    def validate(self) -> None:
        """Valida consistencia topológica. Lanza ValueError con mensaje en castellano."""
        n = len(self.nodes)
        if n < 2 or not self.members:
            raise ValueError("El modelo necesita al menos 2 nodos y 1 barra.")
        if not self.supports:
            raise ValueError("El modelo no tiene apoyos.")
        for k, mem in enumerate(self.members):
            if not (0 <= mem.i < n and 0 <= mem.j < n) or mem.i == mem.j:
                raise ValueError(f"Barra {k}: nodos inválidos ({mem.i}, {mem.j}).")
            if self.member_length(k) < 1e-9:
                raise ValueError(f"Barra {k}: longitud nula.")
            if mem.section.I <= 0 or mem.section.A <= 0 or mem.material.E <= 0:
                raise ValueError(f"Barra {k}: propiedades de sección/material no positivas.")
        seen: set[int] = set()
        for s in self.supports:
            if not 0 <= s.node < n:
                raise ValueError(f"Apoyo en nodo inexistente: {s.node}.")
            if s.node in seen:
                raise ValueError(f"Nodo {s.node} con más de un apoyo.")
            seen.add(s.node)
        for p in self.nodal_loads:
            if not 0 <= p.node < n:
                raise ValueError(f"Carga en nodo inexistente: {p.node}.")
        for q in self.distributed_loads:
            if not 0 <= q.member < len(self.members):
                raise ValueError(f"Carga distribuida en barra inexistente: {q.member}.")
