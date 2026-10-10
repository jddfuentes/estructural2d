"""Renderizado Plotly de la estructura, cargas, deformada y diagramas.

Recibe objetos del core (mm, N, N·mm, MPa) y muestra en unidades de
ingeniería: m, kN, kN/m, kN·m, MPa, mm. No contiene Streamlit: cada función
devuelve un `go.Figure` reutilizable en notebooks o reportes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from core.model import SELF_WEIGHT, DistributedLoad, LoadDirection, Model, SupportType
from core.solver import MemberResult, Results

# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #

C_MEMBER = "#2B3440"
C_SUPPORT = "#5B6573"
C_LOAD = "#C2410C"
C_LOAD_Q = "#EA580C"
C_DEFORMED = "#2563EB"
C_GRID = "rgba(120,130,145,0.18)"
C_REFERENCE = "#475569"
DIAGRAM_COLORS = {"M": "#7C3AED", "V": "#0891B2", "N": "#16A34A", "sigma": "#DC2626", "uy": "#2563EB"}

M_TO = 1e-3  # mm -> m
KN = 1e-3  # N -> kN
KNM = 1e-6  # N·mm -> kN·m


@dataclass(frozen=True, slots=True)
class DiagramSpec:
    title: str
    unit: str
    factor: float  # conversión desde unidades del core


DIAGRAMS: dict[str, DiagramSpec] = {
    "M": DiagramSpec("Momento flector M", "kN·m", KNM),
    "V": DiagramSpec("Esfuerzo de corte V", "kN", KN),
    "N": DiagramSpec("Esfuerzo normal N", "kN", KN),
    "sigma": DiagramSpec("Tensión normal máx. |σ|", "MPa", 1.0),
    "uy": DiagramSpec("Desplazamiento vertical", "mm", 1.0),
}


def _base_layout(fig: go.Figure, title: str, equal_axes: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)),
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        height=480,
    )
    fig.update_xaxes(title_text="x [m]", gridcolor=C_GRID, zeroline=False)
    fig.update_yaxes(title_text="y [m]", gridcolor=C_GRID, zeroline=False)
    if equal_axes:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _extent(model: Model) -> float:
    xy = np.asarray(model.nodes, dtype=float)
    span = xy.max(axis=0) - xy.min(axis=0)
    return float(max(span.max(), 1.0))


def _quantity(mr: MemberResult, q: str) -> np.ndarray:
    return {
        "M": mr.M, "V": mr.V, "N": mr.N, "sigma": mr.sigma_abs, "uy": mr.displacement[:, 1],
    }[q]


# --------------------------------------------------------------------------- #
# Estructura, apoyos, cargas y deformada
# --------------------------------------------------------------------------- #


def plot_structure(
    model: Model,
    results: Results | None = None,
    deformed_scale: float | None = None,
    show_loads: bool = True,
    color_by_stress: bool = True,
) -> go.Figure:
    """Geometría + apoyos + cargas; si hay `results`, deformada amplificada coloreada por |σ|.

    deformed_scale: factor de amplificación; None = automático (δ_max ≈ 8 % del tamaño).
    """
    fig = go.Figure()
    ext = _extent(model)
    s = 0.035 * ext  # tamaño de símbolo [mm]

    # Barras
    xs: list[float | None] = []
    ys: list[float | None] = []
    for mem in model.members:
        (x1, y1), (x2, y2) = model.nodes[mem.i], model.nodes[mem.j]
        xs += [x1 * M_TO, x2 * M_TO, None]
        ys += [y1 * M_TO, y2 * M_TO, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C_MEMBER, width=5),
                             name="Estructura", hoverinfo="skip"))
    nx, ny = zip(*model.nodes, strict=True)
    fig.add_trace(go.Scatter(
        x=np.array(nx) * M_TO, y=np.array(ny) * M_TO, mode="markers",
        marker=dict(size=6, color="white", line=dict(color=C_MEMBER, width=2)),
        name="Nodos", showlegend=False,
        customdata=list(range(len(model.nodes))),
        hovertemplate="Nodo %{customdata}<br>x = %{x:.3f} m<br>y = %{y:.3f} m<extra></extra>",
    ))
    _draw_releases(fig, model, ext)

    for sup in model.supports:
        _draw_support(fig, model, sup.node, sup.type, s)

    if show_loads:
        _draw_distributed_loads(fig, model, ext)
        _draw_point_loads(fig, model, ext)
        if any(d.label == SELF_WEIGHT for d in model.distributed_loads):
            fig.add_annotation(xref="paper", yref="paper", x=0.0, y=0.0, xanchor="left", yanchor="bottom",
                               text="+ peso propio (no dibujado)", showarrow=False,
                               font=dict(size=11, color=C_SUPPORT))

    if results is not None:
        _draw_deformed(fig, model, results, ext, deformed_scale, color_by_stress)

    fig = _base_layout(fig, "Estructura, cargas y deformada" if results else "Estructura y cargas")
    pad = 0.18 * ext * M_TO
    xy = np.asarray(model.nodes) * M_TO
    fig.update_xaxes(range=[xy[:, 0].min() - pad, xy[:, 0].max() + pad])
    _add_reference_axes(fig)
    return fig


def _add_reference_axes(fig: go.Figure) -> None:
    """Añade el sistema global X-Y y la convención del eje Z fuera del plano."""
    arrow_style = dict(
        xref="paper", yref="paper", axref="pixel", ayref="pixel",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor=C_REFERENCE, font=dict(color=C_REFERENCE, size=12),
        bgcolor="rgba(255,255,255,0.86)", borderpad=2,
    )
    fig.add_annotation(x=0.14, y=0.13, ax=-70, ay=0, text="+X", **arrow_style)
    fig.add_annotation(x=0.04, y=0.23, ax=0, ay=70, text="+Y", **arrow_style)
    fig.add_annotation(
        x=0.04, y=0.30, xref="paper", yref="paper", showarrow=False,
        text="<b>Ejes globales</b><br>+Z: sale del plano<br>+Mz: antihorario",
        align="left", font=dict(color=C_REFERENCE, size=11),
        bgcolor="rgba(255,255,255,0.86)", bordercolor=C_REFERENCE, borderwidth=1,
        borderpad=4,
    )


def _member_dir_at(model: Model, node: int) -> tuple[float, float]:
    """Dirección unitaria de la primera barra conectada, saliendo del nodo."""
    for m, mem in enumerate(model.members):
        if node in (mem.i, mem.j):
            c, s = model.member_cos_sin(m)
            return (c, s) if mem.i == node else (-c, -s)
    return (1.0, 0.0)


def _draw_support(fig: go.Figure, model: Model, node: int, kind: SupportType, s: float) -> None:
    x0, y0 = model.nodes[node]
    style = dict(mode="lines", line=dict(color=C_SUPPORT, width=2), hoverinfo="skip", showlegend=False)
    hover = f"Apoyo {kind.value} (nodo {node})"

    if kind is SupportType.FIXED:
        # Muro perpendicular a la barra, del lado opuesto a ella, con rayado
        dx, dy = _member_dir_at(model, node)
        px, py = -dy, dx  # dirección del muro
        a = (x0 + px * s, y0 + py * s)
        b = (x0 - px * s, y0 - py * s)
        wall_x = [a[0], b[0]]
        wall_y = [a[1], b[1]]
        hx: list[float | None] = []
        hy: list[float | None] = []
        for t in np.linspace(-1, 1, 6):
            bx, by = x0 + px * s * t, y0 + py * s * t
            hx += [bx, bx - dx * 0.5 * s + px * 0.35 * s, None]
            hy += [by, by - dy * 0.5 * s + py * 0.35 * s, None]
        fig.add_trace(go.Scatter(x=np.array(wall_x) * M_TO, y=np.array(wall_y) * M_TO,
                                 **{**style, "line": dict(color=C_SUPPORT, width=4)}))
        fig.add_trace(go.Scatter(x=[v * M_TO if v is not None else None for v in hx],
                                 y=[v * M_TO if v is not None else None for v in hy], **style))
    else:
        # Triángulo bajo el nodo (+ línea de rodadura si es móvil) y rayado de suelo
        tri_x = [x0, x0 - 0.7 * s, x0 + 0.7 * s, x0]
        tri_y = [y0, y0 - 1.1 * s, y0 - 1.1 * s, y0]
        fig.add_trace(go.Scatter(x=np.array(tri_x) * M_TO, y=np.array(tri_y) * M_TO,
                                 fill="toself", fillcolor="rgba(91,101,115,0.15)", **style))
        g = y0 - 1.1 * s
        if kind is SupportType.ROLLER:
            fig.add_trace(go.Scatter(
                x=np.array([x0 - 0.35 * s, x0 + 0.35 * s]) * M_TO, y=np.full(2, (g - 0.25 * s) * M_TO),
                mode="markers", marker=dict(size=6, color="white", line=dict(color=C_SUPPORT, width=1.5)),
                hoverinfo="skip", showlegend=False))
            g -= 0.5 * s
        gx: list[float | None] = [x0 - s, x0 + s, None]
        gy: list[float | None] = [g, g, None]
        for t in np.linspace(-1, 0.8, 6):
            gx += [x0 + t * s, x0 + t * s - 0.3 * s, None]
            gy += [g, g - 0.3 * s, None]
        fig.add_trace(go.Scatter(x=[v * M_TO if v is not None else None for v in gx],
                                 y=[v * M_TO if v is not None else None for v in gy], **style))
    # Punto invisible para hover
    fig.add_trace(go.Scatter(x=[x0 * M_TO], y=[y0 * M_TO], mode="markers", marker=dict(size=14, opacity=0),
                             hovertemplate=hover + "<extra></extra>", showlegend=False))


def _draw_releases(
    fig: go.Figure,
    model: Model,
    ext: float,
    results: Results | None = None,
    deformed_scale: float = 0.0,
) -> None:
    """Dibuja rótulas por barra, desplazándolas hacia el interior del elemento."""
    points: list[tuple[float, float]] = []
    inset = 0.018 * ext
    for index, mem in enumerate(model.members):
        (x1, y1), (x2, y2) = model.nodes[mem.i], model.nodes[mem.j]
        if results is None:
            ends = ((x1, y1), (x2, y2))
        else:
            mr = results.members[index]
            ends = (
                (float(mr.points[0, 0] + deformed_scale * mr.displacement[0, 0]),
                 float(mr.points[0, 1] + deformed_scale * mr.displacement[0, 1])),
                (float(mr.points[-1, 0] + deformed_scale * mr.displacement[-1, 0]),
                 float(mr.points[-1, 1] + deformed_scale * mr.displacement[-1, 1])),
            )
        length = math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1])
        if length <= 0:
            continue
        ux = (ends[1][0] - ends[0][0]) / length
        uy = (ends[1][1] - ends[0][1]) / length
        offset = min(inset, 0.08 * length)
        if mem.release_start:
            points.append((ends[0][0] + ux * offset, ends[0][1] + uy * offset))
        if mem.release_end:
            points.append((ends[1][0] - ux * offset, ends[1][1] - uy * offset))
    if points:
        fig.add_trace(go.Scatter(
            x=[p[0] * M_TO for p in points], y=[p[1] * M_TO for p in points],
            mode="markers", marker=dict(symbol="circle-open", size=9, color=C_MEMBER, line=dict(width=2)),
            name="Rótula interna", hovertemplate="Rótula interna<extra></extra>",
        ))


def _arrow(fig: go.Figure, head: tuple[float, float], tail: tuple[float, float], color: str,
           text: str = "", width: float = 2.0) -> None:
    fig.add_annotation(
        x=head[0] * M_TO, y=head[1] * M_TO, ax=tail[0] * M_TO, ay=tail[1] * M_TO,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=width, arrowcolor=color,
        text=text, font=dict(color=color, size=12), standoff=0,
        bgcolor="rgba(255,255,255,0.75)" if text else None,
    )


def _draw_point_loads(fig: go.Figure, model: Model, ext: float) -> None:
    L = 0.14 * ext
    for p in model.nodal_loads:
        x0, y0 = model.nodes[p.node]
        F = math.hypot(p.Fx, p.Fy)
        if F > 0:
            ux, uy = p.Fx / F, p.Fy / F
            label = f"{F * KN:.3g} kN"
            _arrow(fig, (x0, y0), (x0 - ux * L, y0 - uy * L), C_LOAD, label, 2.5)
        if p.Mz:
            r = 0.06 * ext
            sign = 1.0 if p.Mz > 0 else -1.0
            th = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 40) * sign + math.pi / 2
            ax_, ay_ = x0 + r * np.cos(th), y0 + r * np.sin(th)
            fig.add_trace(go.Scatter(x=ax_ * M_TO, y=ay_ * M_TO, mode="lines",
                                     line=dict(color=C_LOAD, width=2.5), hoverinfo="skip", showlegend=False))
            _arrow(fig, (ax_[-1], ay_[-1]), (ax_[-3], ay_[-3]), C_LOAD)
            fig.add_annotation(x=x0 * M_TO, y=(y0 + 1.3 * r) * M_TO, text=f"{abs(p.Mz) * KNM:.3g} kN·m",
                               showarrow=False, font=dict(color=C_LOAD, size=12),
                               bgcolor="rgba(255,255,255,0.75)")


def _load_vector(model: Model, dl: DistributedLoad) -> tuple[float, float]:
    """Dirección unitaria global de la carga (para q > 0)."""
    c, s = model.member_cos_sin(dl.member)
    match dl.direction:
        case LoadDirection.GLOBAL_Y:
            return 0.0, 1.0
        case LoadDirection.GLOBAL_X:
            return 1.0, 0.0
        case LoadDirection.LOCAL_Y:
            return -s, c


def _draw_distributed_loads(fig: go.Figure, model: Model, ext: float) -> None:
    # Sumar cargas por (barra, dirección) sin el peso propio (se indica aparte, no se dibuja)
    acc: dict[tuple[int, LoadDirection], list[float]] = {}
    for d in model.distributed_loads:
        if d.label == SELF_WEIGHT:
            continue
        a = acc.setdefault((d.member, d.direction), [0.0, 0.0])
        a[0] += d.q_start
        a[1] += d.q_end
    loads = [DistributedLoad(m, q1, q2, dr) for (m, dr), (q1, q2) in acc.items() if q1 or q2]
    if not loads:
        return
    qmax = max(max(abs(d.q_start), abs(d.q_end)) for d in loads)
    H = 0.10 * ext  # altura de dibujo para qmax

    # Agrupar tramos contiguos con igual dirección y continuidad de q para rotular una sola vez
    groups: list[list[DistributedLoad]] = []
    for d in sorted(loads, key=lambda d: (d.direction.value, d.member)):
        if groups:
            prev = groups[-1][-1]
            slope_prev = (prev.q_end - prev.q_start) / model.member_length(prev.member)
            slope = (d.q_end - d.q_start) / model.member_length(d.member)
            if (prev.direction is d.direction
                    and model.members[prev.member].j == model.members[d.member].i
                    and math.isclose(prev.q_end, d.q_start, rel_tol=1e-6, abs_tol=1e-9)
                    and math.isclose(slope_prev, slope, rel_tol=1e-6, abs_tol=1e-12)):
                groups[-1].append(d)
                continue
        groups.append([d])

    for group in groups:
        outline_x: list[float] = []
        outline_y: list[float] = []
        for d in group:
            mem = model.members[d.member]
            (x1, y1), (x2, y2) = model.nodes[mem.i], model.nodes[mem.j]
            gx, gy = _load_vector(model, d)
            Lm = model.member_length(d.member)
            n = max(2, int(round(Lm / ext * 14)) + 1)
            for t in np.linspace(0, 1, n):
                qv = d.q_start + (d.q_end - d.q_start) * t
                bx, by = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                h = H * qv / qmax  # con signo
                tail = (bx - gx * h, by - gy * h)
                outline_x.append(tail[0])
                outline_y.append(tail[1])
                if abs(qv) > 0.02 * qmax:
                    _arrow(fig, (bx, by), tail, C_LOAD_Q, width=1.3)
        fig.add_trace(go.Scatter(
            x=np.array(outline_x) * M_TO, y=np.array(outline_y) * M_TO, mode="lines",
            line=dict(color=C_LOAD_Q, width=1.5), hoverinfo="skip", showlegend=False))

        q1, q2 = group[0].q_start, group[-1].q_end
        uniform = all(math.isclose(d.q_start, d.q_end) for d in group) and math.isclose(q1, q2)
        if max(abs(q1), abs(q2)) < 0.05 * qmax:
            continue  # p.ej. peso propio frente a cargas mayores: sin rótulo
        label = f"q = {abs(q1):.3g} kN/m" if uniform else f"q = {abs(q1):.3g} → {abs(q2):.3g} kN/m"
        k = len(outline_x) // 2
        fig.add_annotation(x=outline_x[k] * M_TO, y=outline_y[k] * M_TO, text=label, showarrow=False,
                           yshift=12, font=dict(color=C_LOAD_Q, size=12), bgcolor="rgba(255,255,255,0.75)")


def auto_deformed_scale(model: Model, results: Results) -> float:
    dmax = max(float(mr.deflection.max()) for mr in results.members)
    return 0.08 * _extent(model) / dmax if dmax > 0 else 1.0


def _draw_deformed(fig: go.Figure, model: Model, results: Results, ext: float,
                   scale: float | None, color_by_stress: bool) -> None:
    k = scale if scale is not None else auto_deformed_scale(model, results)
    xs: list[float | None] = []
    ys: list[float | None] = []
    px, py, sig, dtot = [], [], [], []
    for mr in results.members:
        p = mr.points + k * mr.displacement
        xs += list(p[:, 0] * M_TO) + [None]
        ys += list(p[:, 1] * M_TO) + [None]
        px += list(p[:, 0] * M_TO)
        py += list(p[:, 1] * M_TO)
        sig += list(mr.sigma_abs)
        dtot += list(mr.deflection)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C_DEFORMED, width=2, dash="dash"),
                             name=f"Deformada (×{k:,.0f})", hoverinfo="skip"))
    _draw_releases(fig, model, ext, results, k)
    if color_by_stress:
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="markers", name="|σ| [MPa]",
            marker=dict(size=7, color=sig, colorscale="Turbo", cmin=0,
                        colorbar=dict(title=dict(text="|σ| [MPa]"), thickness=12, len=0.8)),
            customdata=np.column_stack([sig, dtot]),
            hovertemplate="|σ| = %{customdata[0]:.1f} MPa<br>δ = %{customdata[1]:.2f} mm<extra></extra>",
        ))


# --------------------------------------------------------------------------- #
# Diagramas de solicitaciones
# --------------------------------------------------------------------------- #


def plot_diagram(model: Model, results: Results, quantity: str, moment_on_tension_side: bool = True
                 ) -> go.Figure:
    """Diagrama de 'M', 'V', 'N', 'sigma' o 'uy'.

    Viga horizontal -> gráfico cartesiano clásico f(x).
    Pórtico -> diagrama dibujado perpendicular a cada barra sobre la geometría.
    moment_on_tension_side: M dibujado del lado traccionado (convención usual en Argentina).
    """
    if model.is_horizontal_beam():
        return _beam_diagram(model, results, quantity, moment_on_tension_side)
    return _frame_diagram(model, results, quantity)


def _beam_diagram(model: Model, results: Results, q: str, tension_side: bool) -> go.Figure:
    spec, color = DIAGRAMS[q], DIAGRAM_COLORS[q]
    x0 = model.nodes[0][0]
    xs = np.concatenate([mr.points[:, 0] for mr in results.members])
    vals = np.concatenate([_quantity(mr, q) for mr in results.members]) * spec.factor
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=(xs - x0) * M_TO, y=vals, mode="lines", fill="tozeroy", line=dict(color=color, width=2.5),
        fillcolor=_rgba(color, 0.18), name=spec.title,
        hovertemplate="x = %{x:.3f} m<br>" + q + " = %{y:.3f} " + spec.unit + "<extra></extra>",
    ))
    # Barra de referencia y apoyos
    fig.add_hline(y=0, line=dict(color=C_MEMBER, width=2))
    for sup in model.supports:
        fig.add_vline(x=(model.nodes[sup.node][0] - x0) * M_TO, line=dict(color=C_GRID, width=1, dash="dot"))
    # Extremos
    reversed_axis = q == "M" and tension_side
    for k in {int(np.argmax(vals)), int(np.argmin(vals))}:
        if abs(vals[k]) > 1e-9:
            up = (vals[k] >= 0) != reversed_axis  # el punto queda arriba del eje en pantalla
            fig.add_annotation(x=(xs[k] - x0) * M_TO, y=vals[k], text=f"{vals[k]:.3g} {spec.unit}",
                               showarrow=True, arrowhead=0, ax=0, ay=-28 if up else 28,
                               font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.8)")
    fig = _base_layout(fig, f"{spec.title} [{spec.unit}]", equal_axes=False)
    fig.update_layout(height=360, showlegend=False)
    fig.update_yaxes(title_text=f"{q} [{spec.unit}]")
    if q == "M" and tension_side:
        fig.update_yaxes(autorange="reversed", title_text="M [kN·m] (+ abajo: tracción inferior)")
    return fig


def _frame_diagram(model: Model, results: Results, q: str) -> go.Figure:
    spec, color = DIAGRAMS[q], DIAGRAM_COLORS[q]
    fig = go.Figure()
    for mem in model.members:
        (x1, y1), (x2, y2) = model.nodes[mem.i], model.nodes[mem.j]
        fig.add_trace(go.Scatter(x=[x1 * M_TO, x2 * M_TO], y=[y1 * M_TO, y2 * M_TO], mode="lines",
                                 line=dict(color=C_MEMBER, width=3), hoverinfo="skip", showlegend=False))
    all_vals = np.concatenate([_quantity(mr, q) for mr in results.members])
    vmax = float(np.abs(all_vals).max()) or 1.0
    scale = 0.12 * _extent(model) / vmax

    for mr in results.members:
        c, s = model.member_cos_sin(mr.member)
        ny_ = np.array([-s, c])  # y local
        v = _quantity(mr, q)
        # M del lado traccionado: M > 0 tracciona y local negativa
        sgn = -1.0 if q == "M" else 1.0
        off = mr.points + np.outer(sgn * v * scale, ny_)
        poly = np.vstack([mr.points[:1], off, mr.points[-1:]])
        fig.add_trace(go.Scatter(
            x=poly[:, 0] * M_TO, y=poly[:, 1] * M_TO, mode="lines", fill="toself",
            fillcolor=_rgba(color, 0.22), line=dict(color=color, width=2), showlegend=False,
            customdata=np.r_[v[:1], v, v[-1:]] * spec.factor,
            hovertemplate=f"Barra {mr.member}<br>{q} = %{{customdata:.3f}} {spec.unit}<extra></extra>",
        ))
        # Rótulos: extremos de barra y máximo interior
        idx = {0, len(v) - 1, int(np.argmax(np.abs(v)))}
        for k in idx:
            if abs(v[k]) > 0.02 * vmax:
                fig.add_annotation(x=off[k, 0] * M_TO, y=off[k, 1] * M_TO, text=f"{v[k] * spec.factor:.3g}",
                                   showarrow=False, font=dict(color=color, size=11),
                                   bgcolor="rgba(255,255,255,0.8)")
    fig = _base_layout(fig, f"{spec.title} [{spec.unit}]")
    fig.update_layout(showlegend=False)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
