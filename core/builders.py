"""Constructores de modelos: viga continua, pórtico simple y pórtico/malla 2D genérico.

`build_beam` / `build_portal_frame` traducen una definición "de ingeniero" (posiciones x a lo
largo de la viga) a un `Model` de nodos y barras. `build_frame` arma un pórtico de topología libre
(nodos (X, Y) explícitos, barras i → j en cualquier orientación, vínculos en cualquier nodo) y
valida la geometría. Sin UI. Unidades: mm, N, N/mm, N·mm.
Signos en ejes globales: Fy < 0 y q < 0 = hacia abajo.

Como el solver es exacto por barra, sólo se crean nodos en puntos singulares
(apoyos, cargas puntuales, inicio/fin de distribuidas): no hace falta mallar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from core.materials import Material, Section
from core.model import (
    SELF_WEIGHT,
    DistributedLoad,
    LoadCase,
    LoadDirection,
    Member,
    Model,
    NodalLoad,
    Spring,
    Support,
    SupportType,
    ThermalLoad,
)

G = 9.80665  # [m/s²]
_TOL = 1e-6  # [mm] tolerancia para unificar posiciones


@dataclass(frozen=True, slots=True)
class BeamSupport:
    x: float  # [mm]
    type: SupportType


@dataclass(frozen=True, slots=True)
class BeamPointLoad:
    x: float  # [mm]
    Fy: float = 0.0  # [N]  (< 0 hacia abajo)
    Fx: float = 0.0  # [N]
    Mz: float = 0.0  # [N·mm] antihorario +
    case: LoadCase = LoadCase.DEAD  # hipótesis de carga (D / L)


@dataclass(frozen=True, slots=True)
class BeamDistLoad:
    x_start: float  # [mm]
    x_end: float  # [mm]
    q_start: float  # [N/mm] (= kN/m), < 0 hacia abajo
    q_end: float  # [N/mm]
    case: LoadCase = LoadCase.DEAD  # hipótesis de carga (D / L)


def self_weight_q(section: Section, material: Material) -> float:
    """Peso propio por unidad de longitud [N/mm], negativo (hacia abajo)."""
    return -material.rho * 1e-9 * section.A * G


def build_beam(
    length: float,
    supports: list[BeamSupport],
    section: Section,
    material: Material,
    point_loads: list[BeamPointLoad] | None = None,
    dist_loads: list[BeamDistLoad] | None = None,
    self_weight: bool = False,
) -> Model:
    """Viga recta horizontal (simple, continua o en voladizo) sobre el eje X."""
    point_loads = point_loads or []
    dist_loads = dist_loads or []
    if length <= 0:
        raise ValueError("La longitud debe ser positiva.")

    def check(x: float, what: str) -> None:
        if not -_TOL <= x <= length + _TOL:
            raise ValueError(f"{what} fuera de la viga: x = {x:g} mm (L = {length:g} mm).")

    for s in supports:
        check(s.x, "Apoyo")
    for p in point_loads:
        check(p.x, "Carga puntual")
    for d in dist_loads:
        check(d.x_start, "Carga distribuida")
        check(d.x_end, "Carga distribuida")
        if d.x_end - d.x_start <= _TOL:
            raise ValueError("Carga distribuida: x_fin debe ser mayor que x_inicio.")

    xs = sorted(
        {0.0, length}
        | {s.x for s in supports}
        | {p.x for p in point_loads}
        | {d.x_start for d in dist_loads}
        | {d.x_end for d in dist_loads}
    )
    xs = _unique(xs)
    idx = {x: i for i, x in enumerate(xs)}

    def node_at(x: float) -> int:
        return idx[min(xs, key=lambda v: abs(v - x))]

    model = Model(nodes=[(x, 0.0) for x in xs])
    model.members = [Member(i, i + 1, section, material) for i in range(len(xs) - 1)]
    model.supports = [Support(node_at(s.x), s.type) for s in supports]
    model.nodal_loads = [
        NodalLoad(node_at(p.x), Fx=p.Fx, Fy=p.Fy, Mz=p.Mz, case=p.case) for p in point_loads
    ]

    for d in dist_loads:
        slope = (d.q_end - d.q_start) / (d.x_end - d.x_start)
        for m in range(len(xs) - 1):
            a, b = xs[m], xs[m + 1]
            if a >= d.x_start - _TOL and b <= d.x_end + _TOL:
                model.distributed_loads.append(
                    DistributedLoad(
                        m,
                        d.q_start + slope * (a - d.x_start),
                        d.q_start + slope * (b - d.x_start),
                        LoadDirection.GLOBAL_Y,
                        case=d.case,
                    )
                )
    if self_weight:
        _add_self_weight(model)
    return model


def build_portal_frame(
    span: float,
    height: float,
    column_section: Section,
    beam_section: Section,
    material: Material,
    base_left: SupportType = SupportType.FIXED,
    base_right: SupportType = SupportType.FIXED,
    beam_q: float = 0.0,
    lateral_load: float = 0.0,
    beam_point_loads: list[BeamPointLoad] | None = None,
    self_weight: bool = False,
    beam_q_case: LoadCase = LoadCase.DEAD,
    lateral_case: LoadCase = LoadCase.DEAD,
) -> Model:
    """Pórtico simple de una nave: 2 columnas + viga, nudos rígidos.

    beam_q: carga distribuida uniforme vertical sobre la viga [N/mm] (< 0 abajo), hipótesis
        `beam_q_case` (D por defecto).
    lateral_load: carga horizontal en el nudo superior izquierdo [N] (> 0 hacia +X), hipótesis
        `lateral_case` (D por defecto). Viento (W) y sismo (E) no están modelados.
    beam_point_loads: cargas puntuales sobre la viga, x medido desde la columna izquierda.
    Nodos: 0 = base izq., 1 = nudo sup. izq., ..., n-2 = nudo sup. der., n-1 = base der.
    """
    if span <= 0 or height <= 0:
        raise ValueError("Luz y altura deben ser positivas.")
    beam_point_loads = beam_point_loads or []
    for p in beam_point_loads:
        if not -_TOL <= p.x <= span + _TOL:
            raise ValueError(f"Carga puntual fuera de la viga: x = {p.x:g} mm.")

    bx = _unique(sorted({0.0, span} | {p.x for p in beam_point_loads}))
    nodes: list[tuple[float, float]] = [(0.0, 0.0)] + [(x, height) for x in bx] + [(span, 0.0)]
    base_l, top_l, top_r, base_r = 0, 1, len(bx), len(bx) + 1

    members = [Member(base_l, top_l, column_section, material)]
    beam_members = list(range(1, len(bx)))
    members += [Member(i, i + 1, beam_section, material) for i in range(1, len(bx))]
    members.append(Member(base_r, top_r, column_section, material))

    model = Model(nodes=nodes, members=members)
    model.supports = [Support(base_l, base_left), Support(base_r, base_right)]
    if lateral_load:
        model.nodal_loads.append(NodalLoad(top_l, Fx=lateral_load, case=lateral_case))
    for p in beam_point_loads:
        n = 1 + min(range(len(bx)), key=lambda k: abs(bx[k] - p.x))
        model.nodal_loads.append(NodalLoad(n, Fx=p.Fx, Fy=p.Fy, Mz=p.Mz, case=p.case))
    if beam_q:
        model.distributed_loads += [
            DistributedLoad(m, beam_q, beam_q, LoadDirection.GLOBAL_Y, case=beam_q_case)
            for m in beam_members
        ]
    if self_weight:
        _add_self_weight(model)
    return model


def build_frame(
    nodes: Sequence[tuple[float, float]],
    members: Sequence[Member],
    supports: Sequence[Support],
    nodal_loads: Sequence[NodalLoad] | None = None,
    distributed_loads: Sequence[DistributedLoad] | None = None,
    thermal_loads: Sequence[ThermalLoad] | None = None,
    springs: Sequence[Spring] | None = None,
    self_weight: bool = False,
) -> Model:
    """Pórtico o malla plana de topología libre.

    nodes: coordenadas globales (X, Y) [mm] de cada nodo; el índice en la lista es su número.
    members: barras `Member(i, j, section, material, release_start, release_end)` con conectividad
        arbitraria e inclinación cualquiera (ejes locales: x de i a j, y local = x rotado +90°).
    supports: vínculos rígidos `Support(node, type)` en cualquier nodo (empotrado, articulado, móvil
        sobre superficie horizontal —restringe uy— o libre); springs: apoyos elásticos `Spring`.
    nodal_loads / distributed_loads / thermal_loads: cargas con su hipótesis (`case`, D por defecto);
        `DistributedLoad.member` y `ThermalLoad.member` refieren al índice de la barra en `members`.
    self_weight: agrega el peso propio (hipótesis D) a todas las barras.

    No subdivide barras: el solver es exacto por barra; las cargas puntuales van en nodos.

    Raises:
        ValueError: coordenadas no finitas, nodos coincidentes, nodos sin barras, barras repetidas
            entre el mismo par de nodos o cualquier inconsistencia de `Model.validate()`.
    """
    pts = [(float(x), float(y)) for x, y in nodes]
    for k, (x, y) in enumerate(pts):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Nodo {k}: coordenadas no finitas ({x}, {y}).")
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if math.hypot(pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]) <= _TOL:
                raise ValueError(
                    f"Nodos {a} y {b} coincidentes en ({pts[a][0]:g}, {pts[a][1]:g}) mm: unificarlos y "
                    "conectar las barras al mismo nodo."
                )
    pairs: dict[frozenset[int], int] = {}
    for k, mem in enumerate(members):
        key = frozenset((mem.i, mem.j))
        if key in pairs and len(key) == 2:
            raise ValueError(
                f"Barras {pairs[key]} y {k} unen los mismos nodos ({mem.i}, {mem.j}): sumar sus "
                "propiedades en una única barra."
            )
        pairs[key] = k
    connected = {n for mem in members for n in (mem.i, mem.j)}
    loose = [k for k in range(len(pts)) if k not in connected]
    if loose and members:
        raise ValueError(
            f"Nodos sin barras: {', '.join(map(str, loose))}. Conectarlos a una barra o eliminarlos."
        )

    model = Model(
        nodes=pts,
        members=list(members),
        supports=list(supports),
        nodal_loads=list(nodal_loads or []),
        distributed_loads=list(distributed_loads or []),
        springs=list(springs or []),
        thermal_loads=list(thermal_loads or []),
    )
    model.validate()
    if self_weight:
        _add_self_weight(model)
    return model


def _add_self_weight(model: Model) -> None:
    for m, mem in enumerate(model.members):
        q = self_weight_q(mem.section, mem.material)
        model.distributed_loads.append(DistributedLoad(m, q, q, LoadDirection.GLOBAL_Y, SELF_WEIGHT))


def _unique(xs: list[float]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if not out or x - out[-1] > _TOL:
            out.append(x)
    return out
