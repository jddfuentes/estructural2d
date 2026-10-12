"""Modelo de datos de la estructura 2D (pórtico plano).

Sistema global: X horizontal a la derecha, Y vertical hacia arriba, giro Z
antihorario positivo. Unidades: mm, N, N/mm, N·mm.

Cada nodo tiene 3 GDL: (ux, uy, rz). Las barras son elementos de pórtico
Euler-Bernoulli con rigidez axial y a flexión, opcionalmente con rótulas en sus extremos.

Vínculos: rígidos (`Support`) y/o elásticos (`Spring`: resortes nodales kx, ky, krz).

Hipótesis de carga: cada carga pertenece a un `LoadCase` (D = permanente, L = sobrecarga de uso;
por defecto D). Las combinaciones normativas (`LoadCombination`) se arman en el solver por
superposición lineal de las respuestas de cada hipótesis.

Cargas térmicas (`ThermalLoad`): variación uniforme de temperatura ΔT [°C] en una barra (y,
opcionalmente, una deformación axial impuesta adicional ε_imp [-]). Deformación axial libre
ε₀ = α·ΔT + ε_imp; sólo produce esfuerzos si la estructura impide el alargamiento.
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
    FREE = "libre"  # no restringe ningún GDL; permite apoyos elásticos puros

    @property
    def restrained_dofs(self) -> tuple[bool, bool, bool]:
        return {
            SupportType.FIXED: (True, True, True),
            SupportType.PINNED: (True, True, False),
            SupportType.ROLLER: (False, True, False),
            SupportType.FREE: (False, False, False),
        }[self]


class LoadCase(str, Enum):
    """Hipótesis (estado) de carga. Nomenclatura de CIRSOC 101-2005 / ASCE 7 (§2.3)."""

    DEAD = "D"  # cargas permanentes: peso propio, terminaciones, equipos fijos
    LIVE = "L"  # sobrecarga de uso (ocupación, mantenimiento, cargas móviles)


class LoadDirection(str, Enum):
    LOCAL_Y = "local_y"  # perpendicular a la barra (y local)
    GLOBAL_Y = "global_y"  # vertical global, por unidad de longitud de barra
    GLOBAL_X = "global_x"  # horizontal global, por unidad de longitud de barra


@dataclass(frozen=True, slots=True)
class Member:
    """Barra de pórtico Euler-Bernoulli del nodo `i` al nodo `j`.

    Rótulas internas: `release_start` / `release_end` liberan el momento flector Mz en el
    extremo i / j de la barra (M = 0 en ese extremo; el giro de la barra queda desacoplado
    del giro del nudo). Por defecto ambas en False: unión rígida continua (comportamiento
    previo). Con ambas en True la barra es una biela: a los nudos sólo transmite axil y el
    corte isostático de su propia carga transversal (trabaja como viga simplemente apoyada).
    """

    i: int  # nodo inicial
    j: int  # nodo final
    section: Section
    material: Material
    release_start: bool = False  # rótula (Mz = 0) en el nodo i
    release_end: bool = False  # rótula (Mz = 0) en el nodo j


@dataclass(frozen=True, slots=True)
class Support:
    node: int
    type: SupportType


@dataclass(frozen=True, slots=True)
class Spring:
    """Apoyo elástico: resortes nodales desacoplados según los ejes globales.

    Cada resorte actúa sobre la estructura con R = −k·u en su GDL (fuerza o momento opuesto al
    desplazamiento o giro del nodo). k = 0 ⇒ sin resorte en ese GDL. Un GDL con resorte y sin
    restricción rígida es un GDL libre: entra al sistema de ecuaciones.
    Si el nodo además tiene un `Support`, en los GDL restringidos rígidamente el resorte no
    trabaja (u = 0) y la reacción es la del vínculo rígido (p. ej. articulado + krz = empotramiento
    elástico).
    """

    node: int
    kx: float = 0.0  # [N/mm] rigidez traslacional según X global
    ky: float = 0.0  # [N/mm] rigidez traslacional según Y global
    krz: float = 0.0  # [N·mm/rad] rigidez rotacional (giro Z, antihorario +)

    def __post_init__(self) -> None:
        for name, k in (("kx", self.kx), ("ky", self.ky), ("krz", self.krz)):
            if not (math.isfinite(k) and k >= 0.0):
                raise ValueError(
                    f"Resorte en nodo {self.node}: {name} = {k} no es válida; la rigidez debe ser "
                    "un número finito ≥ 0 (kx, ky en N/mm; krz en N·mm/rad)."
                )

    @property
    def stiffness(self) -> tuple[float, float, float]:
        """(kx [N/mm], ky [N/mm], krz [N·mm/rad]) en el orden de los GDL (ux, uy, rz)."""
        return (self.kx, self.ky, self.krz)

    def inert(self, support: SupportType | None) -> tuple[str, ...]:
        """Rigideces > 0 ("kx", "ky", "krz") sobre GDL que `support` ya restringe: no trabajan (u = 0)."""
        if support is None:
            return ()
        names = ("kx", "ky", "krz")
        return tuple(n for n, k, r in zip(names, self.stiffness, support.restrained_dofs, strict=True)
                     if k > 0.0 and r)

    def active(self, support: SupportType | None) -> Spring:
        """Copia con k = 0 en los GDL restringidos por `support` (mismo resultado de cálculo).

        Sirve para mostrar/documentar sólo los resortes que trabajan (UI, PDF).
        """
        r = support.restrained_dofs if support is not None else (False, False, False)
        kx, ky, krz = (0.0 if rr else k for k, rr in zip(self.stiffness, r, strict=True))
        return Spring(self.node, kx, ky, krz)

    @property
    def is_null(self) -> bool:
        """True si las tres rigideces son nulas."""
        return self.kx == 0.0 and self.ky == 0.0 and self.krz == 0.0


@dataclass(frozen=True, slots=True)
class NodalLoad:
    """Carga concentrada en nodo, ejes globales. Fy < 0 = hacia abajo."""

    node: int
    Fx: float = 0.0  # [N]
    Fy: float = 0.0  # [N]
    Mz: float = 0.0  # [N·mm], antihorario +
    case: LoadCase = LoadCase.DEAD  # hipótesis de carga (acepta también "D" / "L")

    def __post_init__(self) -> None:
        object.__setattr__(self, "case", _as_load_case(self.case))


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
    case: LoadCase = LoadCase.DEAD  # hipótesis de carga (acepta también "D" / "L")

    def __post_init__(self) -> None:
        object.__setattr__(self, "case", _as_load_case(self.case))


@dataclass(frozen=True, slots=True)
class ThermalLoad:
    """Variación uniforme de temperatura ΔT en una barra (+ deformación axial impuesta opcional).

    Deformación axial libre (sin tensión) de la barra: ε₀ = α·ΔT + `strain`, con α del material de
    la barra [1/°C]. ΔT > 0 dilata (alarga), ΔT < 0 contrae. `strain` [-] es una deformación axial
    impuesta adicional (p. ej. falta de ajuste: ε = ΔL/L; > 0 barra más larga que la distancia entre
    nudos). Uniforme en la sección: no hay gradiente térmico (no induce curvatura propia).
    """

    member: int
    delta_t: float  # [°C] variación uniforme de temperatura (> 0 dilatación)
    case: LoadCase = LoadCase.DEAD  # hipótesis de carga (acepta también "D" / "L")
    strain: float = 0.0  # [-] deformación axial impuesta adicional (alargamiento > 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case", _as_load_case(self.case))
        for name, val in (("ΔT", self.delta_t), ("deformación impuesta", self.strain)):
            if not math.isfinite(val):
                raise ValueError(
                    f"Carga térmica en barra {self.member}: {name} = {val} no es un número finito."
                )

    def imposed_strain(self, alpha: float) -> float:
        """ε₀ = α·ΔT + ε_imp [-]: alargamiento unitario libre de la barra (α [1/°C])."""
        return alpha * self.delta_t + self.strain


def _as_load_case(case: LoadCase | str) -> LoadCase:
    """Normaliza la hipótesis de carga ("D" / "L" o `LoadCase`); ValueError si no existe."""
    try:
        return LoadCase(case)
    except ValueError:
        valid = ", ".join(f"{c.value!r}" for c in LoadCase)
        raise ValueError(f"Hipótesis de carga desconocida: {case!r}. Valores válidos: {valid}.") from None


# Estados límite de una combinación (rótulos en castellano, se muestran tal cual en UI y PDF)
ULS = "ELU"  # estado límite último (resistencia), cargas mayoradas
SLS = "ELS"  # estado límite de servicio (flechas), cargas sin mayorar


@dataclass(frozen=True, slots=True)
class LoadCombination:
    """Combinación lineal de hipótesis: γ_D·D + γ_L·L.

    `limit_state`: "ELU" (resistencia, cargas mayoradas) o "ELS" (servicio, cargas de servicio).
    """

    name: str  # p. ej. "1.2D + 1.6L"
    gamma_D: float
    gamma_L: float
    limit_state: str = ULS

    def __post_init__(self) -> None:
        if self.limit_state not in (ULS, SLS):
            raise ValueError(
                f"Combinación {self.name!r}: estado límite {self.limit_state!r} inválido; "
                f"usar {ULS!r} (resistencia) o {SLS!r} (servicio)."
            )
        for g_name, g in (("γ_D", self.gamma_D), ("γ_L", self.gamma_L)):
            if not (math.isfinite(g) and g >= 0.0):
                raise ValueError(
                    f"Combinación {self.name!r}: {g_name} = {g} no es válido; debe ser un número finito ≥ 0."
                )

    def factor(self, case: LoadCase) -> float:
        """Factor de carga γ de la hipótesis `case` en esta combinación."""
        return self.gamma_D if case is LoadCase.DEAD else self.gamma_L


# Combinaciones estándar. Fuente: CIRSOC 301-2005 §A.4.2, ecs. (A.4-1) y (A.4-2), y AISC 360-16
# §B2 (remite a ASCE/SEI 7 §2.3), combinaciones LRFD sin cargas de cubierta (Lr), nieve (S),
# lluvia (R), viento (W) ni sismo (E). Servicio: AISC 360-16 cap. L y su comentario
# (combinación de servicio D + L, cargas sin mayorar).
COMB_1_4D = LoadCombination("1.4D", 1.4, 0.0, ULS)
COMB_1_2D_1_6L = LoadCombination("1.2D + 1.6L", 1.2, 1.6, ULS)
COMB_SERVICE_D_L = LoadCombination("1.0D + 1.0L", 1.0, 1.0, SLS)
ULS_COMBINATIONS: tuple[LoadCombination, ...] = (COMB_1_4D, COMB_1_2D_1_6L)


SELF_WEIGHT = "peso propio"


@dataclass(slots=True)
class Model:
    nodes: list[tuple[float, float]] = field(default_factory=list)
    members: list[Member] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    nodal_loads: list[NodalLoad] = field(default_factory=list)
    distributed_loads: list[DistributedLoad] = field(default_factory=list)
    springs: list[Spring] = field(default_factory=list)  # apoyos elásticos (resortes nodales)
    thermal_loads: list[ThermalLoad] = field(default_factory=list)  # ΔT / deformación axial impuesta

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

    def load_cases(self) -> tuple[LoadCase, ...]:
        """Hipótesis con al menos una carga, en el orden de `LoadCase` (D primero)."""
        used = (
            {p.case for p in self.nodal_loads}
            | {q.case for q in self.distributed_loads}
            | {t.case for t in self.thermal_loads}
        )
        return tuple(c for c in LoadCase if c in used)

    def has_live_loads(self) -> bool:
        """True si alguna carga pertenece a la hipótesis L (sobrecarga de uso)."""
        return LoadCase.LIVE in self.load_cases()

    def is_horizontal_beam(self) -> bool:
        """True si todos los nodos están sobre y = cte (viga continua)."""
        ys = {round(y, 9) for _, y in self.nodes}
        return len(ys) == 1

    def validate(self) -> None:
        """Valida consistencia topológica. Lanza ValueError con mensaje en castellano."""
        n = len(self.nodes)
        if n < 2 or not self.members:
            raise ValueError("El modelo necesita al menos 2 nodos y 1 barra.")
        if not self.supports and not self.springs:
            raise ValueError("El modelo no tiene apoyos (ni rígidos ni elásticos).")
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
        seen_springs: set[int] = set()
        for sp in self.springs:
            if not 0 <= sp.node < n:
                raise ValueError(f"Resorte en nodo inexistente: {sp.node}.")
            if sp.node in seen_springs:
                raise ValueError(
                    f"Nodo {sp.node} con más de un resorte: definir kx, ky y krz en un único `Spring`."
                )
            seen_springs.add(sp.node)
        for p in self.nodal_loads:
            if not 0 <= p.node < n:
                raise ValueError(f"Carga en nodo inexistente: {p.node}.")
        for q in self.distributed_loads:
            if not 0 <= q.member < len(self.members):
                raise ValueError(f"Carga distribuida en barra inexistente: {q.member}.")
        for t in self.thermal_loads:
            if not 0 <= t.member < len(self.members):
                raise ValueError(f"Carga térmica en barra inexistente: {t.member}.")
