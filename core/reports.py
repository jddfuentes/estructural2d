"""Memoria de cálculo en PDF (A4) con ReportLab.

Capa de presentación del core (ver AGENTS.md §3/§4): recibe objetos del core en
unidades SI-mm y los MUESTRA en unidades de ingeniería (m, kN, kN/m, kN·m, cm², cm⁴,
cm³). Las conversiones viven sólo en este módulo y nunca vuelven al resto del core.

Función pública:
    build_pdf_report(model, results, check, project_title, author, *, date, reference_length) -> bytes

Pura: no lee ni escribe archivos, no muta sus argumentos y, con `date` fijo, devuelve
exactamente los mismos bytes (PDF "invariant": sin marcas de tiempo ni ID aleatorio).
"""

from __future__ import annotations

import datetime as dt
import io
import math
import re
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from core.glossary import GRAPHIC_SYMBOLS, NOTATION, TERMS
from core.materials import Material, Section
from core.model import SELF_WEIGHT, DistributedLoad, LoadDirection, Model
from core.solver import Results
from core.verification import SafetyCheck, Status

__all__ = ["build_pdf_report"]

APP_NAME = "Estructural 2D"

# --------------------------------------------------------------------------- #
# Unidades de presentación (único lugar del core donde se convierte, sólo para mostrar)
# --------------------------------------------------------------------------- #

_M = 1e-3  # mm -> m
_KN = 1e-3  # N -> kN
_KNM = 1e-6  # N·mm -> kN·m
_CM2, _CM3, _CM4 = 1e-2, 1e-3, 1e-4  # mm² -> cm², mm³ -> cm³, mm⁴ -> cm⁴

# --------------------------------------------------------------------------- #
# Estilo (sobrio, monocromo con un acento)
# --------------------------------------------------------------------------- #

INK = colors.HexColor("#1F2933")
ACCENT = colors.HexColor("#1F3A5F")
MUTED = colors.HexColor("#5F6B7A")
RULE = colors.HexColor("#9AA5B1")
HEAD_BG = colors.HexColor("#E6EBF1")
ZEBRA = colors.HexColor("#F6F8FA")
STATUS_COLORS: dict[Status, tuple[colors.Color, colors.Color]] = {
    Status.OK: (colors.HexColor("#14532D"), colors.HexColor("#DCFCE7")),
    Status.ALERT: (colors.HexColor("#78350F"), colors.HexColor("#FEF3C7")),
    Status.FAIL: (colors.HexColor("#7F1D1D"), colors.HexColor("#FEE2E2")),
}
VERDICT_TEXT: dict[Status, str] = {
    Status.OK: "OK — VERIFICA",
    Status.ALERT: "ALERTA — FS menor al admisible",
    Status.FAIL: "FLUENCIA — σ<sub>máx</sub> supera Sy",
}

PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 20 * mm, 24 * mm, 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

_FONT, _BOLD = "Helvetica", "Helvetica-Bold"

STYLES: dict[str, ParagraphStyle] = {
    "title": ParagraphStyle("title", fontName=_BOLD, fontSize=17, leading=21, textColor=ACCENT, spaceAfter=2),
    "subtitle": ParagraphStyle("subtitle", fontName=_FONT, fontSize=9.5, leading=12, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName=_BOLD, fontSize=11.5, leading=14, textColor=ACCENT,
                         spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName=_BOLD, fontSize=9.5, leading=12, textColor=INK,
                         spaceBefore=6, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName=_FONT, fontSize=9, leading=12.2, textColor=INK),
    "note": ParagraphStyle("note", fontName=_FONT, fontSize=7.8, leading=10, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName=_FONT, fontSize=8.2, leading=10, textColor=INK),
    "cell_r": ParagraphStyle("cell_r", fontName=_FONT, fontSize=8.2, leading=10, textColor=INK,
                             alignment=TA_RIGHT),
    "cell_h": ParagraphStyle("cell_h", fontName=_BOLD, fontSize=8, leading=10, textColor=INK),
    "cell_hr": ParagraphStyle("cell_hr", fontName=_BOLD, fontSize=8, leading=10, textColor=INK,
                              alignment=TA_RIGHT),
    "meta_k": ParagraphStyle("meta_k", fontName=_BOLD, fontSize=8, leading=10, textColor=MUTED),
    "meta_v": ParagraphStyle("meta_v", fontName=_FONT, fontSize=9, leading=11, textColor=INK),
}


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #


def build_pdf_report(
    model: Model,
    results: Results,
    check: SafetyCheck,
    project_title: str = "Memoria de Cálculo",
    author: str = "",
    *,
    date: dt.date | None = None,
    reference_length: float | None = None,
) -> bytes:
    """Genera la memoria de cálculo y devuelve el PDF como bytes (empieza con b"%PDF").

    Args:
        model, results, check: salida de `solve()` y `check_safety()` para el mismo modelo.
        project_title, author: texto libre del encabezado.
        date: fecha del cálculo (por defecto, hoy). Fijarla hace la salida reproducible.
        reference_length: longitud de referencia para la flecha relativa L/δ [mm]
            (por defecto, la extensión horizontal de la estructura).
    """
    date = date or dt.date.today()
    title = project_title.strip() or "Memoria de Cálculo"
    author = author.strip()
    L_ref = reference_length if reference_length else _horizontal_extent(model)

    buf = io.BytesIO()
    doc = _Doc(buf, title=title, author=author or APP_NAME, date=date)
    story: list[Flowable] = []
    story += _title_block(model, title, author, date)
    story += _section_basis(check)
    story += _section_inputs(model)
    story += _section_results(model, results, L_ref)
    story += _section_verification(check)
    story += _section_limitations()
    doc.build(story, canvasmaker=_NumberedCanvas)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Documento, encabezado y pie
# --------------------------------------------------------------------------- #


class _Doc(BaseDocTemplate):
    def __init__(self, buf: io.BytesIO, title: str, author: str, date: dt.date) -> None:
        super().__init__(
            buf, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOT,
            title=title, author=author, subject="Memoria de cálculo estructural 2D",
            creator=APP_NAME, invariant=1, lang="es-AR",
        )
        self.report_title = title
        self.report_date = date
        frame = Frame(MARGIN_X, MARGIN_BOT, CONTENT_W, PAGE_H - MARGIN_TOP - MARGIN_BOT,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canv: Canvas, doc: BaseDocTemplate) -> None:
        canv.saveState()
        top = PAGE_H - 13 * mm
        canv.setFont(_BOLD, 7.5)
        canv.setFillColor(ACCENT)
        canv.drawString(MARGIN_X, top, "MEMORIA DE CÁLCULO ESTRUCTURAL")
        canv.setFont(_FONT, 7.5)
        canv.setFillColor(MUTED)
        canv.drawRightString(PAGE_W - MARGIN_X, top, _clip(self.report_title, 70))
        canv.setStrokeColor(ACCENT)
        canv.setLineWidth(0.8)
        canv.line(MARGIN_X, top - 2.5 * mm, PAGE_W - MARGIN_X, top - 2.5 * mm)
        bottom = 11 * mm
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN_X, bottom + 4 * mm, PAGE_W - MARGIN_X, bottom + 4 * mm)
        canv.drawString(MARGIN_X, bottom,
                        f"{APP_NAME} · cálculo elástico lineal 2D · {self.report_date:%d/%m/%Y}")
        canv.restoreState()


class _NumberedCanvas(Canvas):
    """Canvas que difiere el pie para poder escribir 'Página n de N'."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pages: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802  (API de ReportLab)
        self._pages.append(dict(self.__dict__))
        self._startPage()  # type: ignore[attr-defined]  # receta oficial de ReportLab

    def save(self) -> None:
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self.saveState()
            self.setFont(_FONT, 7.5)
            self.setFillColor(MUTED)
            self.drawRightString(PAGE_W - MARGIN_X, 11 * mm, f"Página {self.getPageNumber()} de {total}")
            self.restoreState()
            super().showPage()
        super().save()


# --------------------------------------------------------------------------- #
# Secciones del documento
# --------------------------------------------------------------------------- #


def _title_block(model: Model, title: str, author: str, date: dt.date) -> list[Flowable]:
    kind = "Viga recta" if model.is_horizontal_beam() else "Pórtico plano"
    meta = [
        ("PROYECTO", escape(title)),
        ("AUTOR", escape(author) or "—"),
        ("FECHA", f"{date:%d/%m/%Y}"),
        ("ESTRUCTURA", f"{kind} · {len(model.nodes)} nodos · {len(model.members)} barras"),
        ("MÉTODO", "Rigidez directa · Euler-Bernoulli"),
        ("SOFTWARE", APP_NAME),
    ]
    cells = [[_p(k, "meta_k"), _p(v, "meta_v")] for k, v in meta]
    rows = [cells[i] + cells[i + 1] for i in range(0, len(cells), 2)]
    w = CONTENT_W / 2
    t = Table(rows, colWidths=[0.33 * w, 0.67 * w, 0.33 * w, 0.67 * w])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, ACCENT),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
        ("BACKGROUND", (0, 0), (0, -1), HEAD_BG),
        ("BACKGROUND", (2, 0), (2, -1), HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [
        _p(escape(title), "title"),
        _p("Memoria de cálculo — predimensionamiento estructural 2D", "subtitle"),
        Spacer(1, 5 * mm),
        t,
        Spacer(1, 2 * mm),
    ]


def _section_basis(check: SafetyCheck) -> list[Flowable]:
    items = [
        "Análisis elástico lineal de primer orden por el método directo de rigidez; elementos de "
        "pórtico plano Euler-Bernoulli (sin deformación por corte). Solicitaciones y elástica exactas "
        "por barra para cargas lineales.",
        "Unidades: longitudes en m (secciones en mm/cm), fuerzas en kN, momentos en kN·m, "
        "tensiones en MPa.",
        "Ejes globales: X hacia la derecha, Y hacia arriba, momentos positivos antihorarios. "
        "Solicitaciones: N &gt; 0 tracción; M &gt; 0 tracciona la fibra inferior (y local negativa).",
        f"Criterio de verificación: tensión normal elástica en fibra extrema "
        f"{_g('s')} = N/A ± M·c/I; FS = S<sub>y</sub> / {_g('s')}<sub>máx</sub> "
        f"≥ FS<sub>adm</sub> = {check.fs_min:g}.",
    ]
    return [_h1("1. Bases de cálculo"), *(_bullet(t) for t in items)]


def _section_inputs(model: Model) -> list[Flowable]:
    out: list[Flowable] = [_h1("2. Datos de entrada")]

    # 2.1 Geometría
    out.append(_h2("2.1 Geometría"))
    if model.is_horizontal_beam():
        geo = f"Viga recta horizontal de longitud total <b>L = {_horizontal_extent(model) * _M:.3f} m</b>."
    else:
        xs = [x for x, _ in model.nodes]
        ys = [y for _, y in model.nodes]
        geo = (f"Pórtico plano de {(max(xs) - min(xs)) * _M:.3f} m de ancho y "
               f"{(max(ys) - min(ys)) * _M:.3f} m de alto.")
    out.append(_p(geo, "body"))
    out.append(Spacer(1, 2 * mm))
    rows = []
    for m, mem in enumerate(model.members):
        (x1, y1), (x2, y2) = model.nodes[mem.i], model.nodes[mem.j]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        rows.append([str(m), f"{mem.i} → {mem.j}", f"({x1 * _M:.3f}; {y1 * _M:.3f})",
                     f"({x2 * _M:.3f}; {y2 * _M:.3f})", f"{model.member_length(m) * _M:.3f}",
                     f"{ang:.1f}", escape(mem.section.name)])
    out.append(_table(["Barra", "Nodos", "Inicio (x; y) [m]", "Fin (x; y) [m]", "L [m]", "Áng. [°]",
                       "Sección"], rows, [0.08, 0.10, 0.18, 0.18, 0.10, 0.10, 0.26], num_cols={4, 5}))

    # 2.2 Apoyos
    restr = {"empotrado": "ux, uy, θz", "articulado": "ux, uy", "móvil": "uy"}
    rows = [[str(s.node), f"{model.nodes[s.node][0] * _M:.3f}", f"{model.nodes[s.node][1] * _M:.3f}",
             s.type.value.capitalize(), restr.get(s.type.value, "—")] for s in model.supports]
    out.append(KeepTogether([_h2("2.2 Condiciones de apoyo"),
                             _table(["Nodo", "x [m]", "y [m]", "Tipo", "GDL restringidos"], rows,
                                    [0.10, 0.15, 0.15, 0.25, 0.35], num_cols={1, 2})]))

    # 2.3 Cargas
    out.append(_h2("2.3 Cargas aplicadas"))
    out.append(_p("Valores con signo en ejes globales (Fy &lt; 0 y q &lt; 0: hacia abajo).", "note"))
    if model.nodal_loads:
        rows = [[str(p.node), f"{model.nodes[p.node][0] * _M:.3f}", f"{model.nodes[p.node][1] * _M:.3f}",
                 _num(p.Fx * _KN), _num(p.Fy * _KN), _num(p.Mz * _KNM)] for p in model.nodal_loads]
        out.append(_p("<b>Cargas concentradas</b>", "body"))
        out.append(_table(["Nodo", "x [m]", "y [m]", "Fx [kN]", "Fy [kN]", "Mz [kN·m]"], rows,
                          [0.12, 0.16, 0.16, 0.18, 0.18, 0.20], num_cols={1, 2, 3, 4, 5}))
    groups = _group_distributed(model)
    if groups:
        out.append(Spacer(1, 2 * mm))
        out.append(_p("<b>Cargas distribuidas</b>", "body"))
        beam = model.is_horizontal_beam()
        span_head = "Tramo x [m]" if beam else "Desde → hasta (x; y) [m]"
        rows = [[g.origin, g.members,
                 f"{g.p0[0] * _M:.3f} → {g.p1[0] * _M:.3f}" if beam else
                 f"({g.p0[0] * _M:.2f}; {g.p0[1] * _M:.2f}) → ({g.p1[0] * _M:.2f}; {g.p1[1] * _M:.2f})",
                 _DIR_TXT[g.direction], _num(g.q0), _num(g.q1)] for g in groups]  # N/mm == kN/m
        out.append(_table(["Origen", "Barras", span_head, "Dirección", "q ini [kN/m]", "q fin [kN/m]"],
                          rows, [0.14, 0.09, 0.29, 0.18, 0.15, 0.15], num_cols={4, 5}))
    if not model.nodal_loads and not groups:
        out.append(_p("Sin cargas aplicadas.", "body"))

    # 2.4 Materiales y secciones
    out.append(_h2("2.4 Material y sección transversal"))
    mats: dict[str, Material] = {}
    secs: dict[tuple[str, str], tuple[Section, Material]] = {}
    for mem in model.members:
        mats.setdefault(mem.material.name, mem.material)
        secs.setdefault((mem.section.name, mem.material.name), (mem.section, mem.material))
    rows = [[escape(m.name), f"{m.E:,.0f}", f"{m.Sy:g}", f"{m.Su:g}", f"{m.nu:g}", f"{m.rho:,.0f}"]
            for m in mats.values()]
    out.append(_table(["Material", "E [MPa]", "Sy [MPa]", "Su [MPa]", f"{_g('n')} [-]",
                       f"{_g('r')} [kg/m{_sup(3)}]"],
                      rows, [0.30, 0.15, 0.13, 0.13, 0.12, 0.17], num_cols={1, 2, 3, 4, 5}))
    out.append(Spacer(1, 2 * mm))
    rows = [[escape(s.name), f"{s.h:g}", f"{s.A * _CM2:,.2f}", f"{s.I * _CM4:,.1f}", f"{s.W * _CM3:,.1f}",
             f"{s.c_max:g}", f"{s.A * 1e-6 * mat.rho:.2f}"] for s, mat in secs.values()]
    out.append(_table(["Sección", "h [mm]", f"A [cm{_sup(2)}]", f"I [cm{_sup(4)}]", f"W [cm{_sup(3)}]",
                       "c<sub>máx</sub> [mm]", "Masa [kg/m]"],
                      rows, [0.28, 0.10, 0.12, 0.14, 0.12, 0.12, 0.12], num_cols={1, 2, 3, 4, 5, 6}))
    return out


def _section_results(model: Model, results: Results, L_ref: float) -> list[Flowable]:
    out: list[Flowable] = []

    # 3.1 Reacciones + equilibrio global
    rows, sx, sy = [], 0.0, 0.0
    for s in model.supports:
        Rx, Ry, Mz = results.reactions[s.node]
        sx, sy = sx + Rx, sy + Ry
        rows.append([str(s.node), s.type.value.capitalize(), _num(Rx * _KN), _num(Ry * _KN),
                     _num(Mz * _KNM) if s.type.restrained_dofs[2] else "—"])
    rows.append(["<b>Σ</b>", "", f"<b>{_num(sx * _KN)}</b>", f"<b>{_num(sy * _KN)}</b>", ""])
    Fx, Fy = _applied_resultant(model)
    scale = max(abs(Fx), abs(Fy), 1.0)
    out.append(KeepTogether([
        _h1("3. Resultados"),  # dentro del bloque para que el título no quede huérfano
        _h2("3.1 Reacciones de vínculo"),
        _table(["Nodo", "Tipo", "Rx [kN]", "Ry [kN]", "Mz [kN·m]"], rows,
               [0.12, 0.28, 0.20, 0.20, 0.20], num_cols={2, 3, 4}, total_row=True),
        _p(f"Equilibrio global — cargas aplicadas: ΣFx = {_num(Fx * _KN)} kN, ΣFy = {_num(Fy * _KN)} kN. "
           f"Residuo relativo |ΣR + ΣF| / |F| = {max(abs(sx + Fx), abs(sy + Fy)) / scale:.1e}.", "note"),
    ]))

    # 3.2 Solicitaciones máximas
    spec = [("Momento flector M", "M", _KNM, "kN·m"), ("Esfuerzo de corte V", "V", _KN, "kN"),
            ("Esfuerzo normal N", "N", _KN, "kN")]
    rows = []
    for label, q, f, unit in spec:
        e = results.extreme(q)
        rows.append([label, f"{e.value * f:,.3f}", unit, str(e.member),
                     f"({e.point[0] * _M:.3f}; {e.point[1] * _M:.3f})"])
    out.append(KeepTogether([
        _h2("3.2 Solicitaciones máximas"),
        _table(["Solicitación", "Valor máx.", "Unidad", "Barra", "Ubicación (x; y) [m]"], rows,
               [0.30, 0.17, 0.12, 0.11, 0.30], num_cols={1}),
        _p("Valor con signo del máximo en valor absoluto de toda la estructura.", "note"),
    ]))

    # 3.3 Deformación
    d = results.extreme("deflection")
    rel = f"L/{L_ref / abs(d.value):,.0f}" if abs(d.value) > 1e-12 else "—"
    rows = [
        [f"Desplazamiento máximo {_g('d')}<sub>máx</sub>", f"{abs(d.value):,.2f} mm"],
        ["Ubicación (x; y)", f"({d.point[0] * _M:.3f}; {d.point[1] * _M:.3f}) m · barra {d.member}"],
        ["Longitud de referencia L", f"{L_ref * _M:.3f} m"],
        [f"Flecha relativa L/{_g('d')}", f"<b>{rel}</b>"],
    ]
    out.append(KeepTogether([_h2("3.3 Deformación"), _kv_table(rows)]))
    return out


def _section_verification(check: SafetyCheck) -> list[Flowable]:
    fg, bg = STATUS_COLORS[check.status]
    fs_txt = "∞" if math.isinf(check.fs) else f"{check.fs:.2f}"
    rows = [
        ["Tensión de fluencia S<sub>y</sub>", f"{check.sy:g} MPa"],
        [f"Tensión normal máxima |{_g('s')}|<sub>máx</sub>", f"{check.sigma_max:,.1f} MPa"],
        ["Ubicación crítica",
         f"barra {check.member}, ({check.point[0] * _M:.3f}; {check.point[1] * _M:.3f}) m"],
        [f"Factor de seguridad FS = S<sub>y</sub> / {_g('s')}<sub>máx</sub>", f"<b>{fs_txt}</b>"],
        ["Factor de seguridad admisible FS<sub>adm</sub>", f"{check.fs_min:g}"],
        [f"Aprovechamiento {_g('s')}<sub>máx</sub> · FS<sub>adm</sub> / S<sub>y</sub>",
         f"{check.utilization:.0%}"],
    ]
    v_style = ParagraphStyle("verdict", parent=STYLES["body"], fontSize=11, leading=14, textColor=fg)
    verdict = Table([[Paragraph(f"<b>VEREDICTO: {VERDICT_TEXT[check.status]}</b>", v_style)]],
                    colWidths=[CONTENT_W])
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 4, fg),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return [KeepTogether([_h1("4. Verificación de seguridad"), _kv_table(rows), Spacer(1, 3 * mm), verdict])]


def _section_limitations() -> list[Flowable]:
    items = [
        "No se verifican pandeo flexional, pandeo lateral-torsional, abolladura local, tensiones de "
        "corte ni combinadas (von Mises), fatiga ni efectos de segundo orden.",
        "Propiedades de sección nominales de catálogo; verificar contra el catálogo del proveedor.",
        "Documento de predimensionamiento: no reemplaza la memoria de cálculo reglamentaria "
        "(CIRSOC 301 / AISC 360) firmada por profesional matriculado.",
    ]
    return [_h1("5. Limitaciones y alcance"), *(_bullet(t, "note") for t in items)]


# --------------------------------------------------------------------------- #
# 6. Glosario y simbología
# --------------------------------------------------------------------------- #


def _icon(kind: str, w: float = 54.0, h: float = 26.0) -> Drawing:
    """Ícono de la simbología gráfica, dibujado con las mismas primitivas que las figuras."""
    d = Drawing(w, h)
    mid = w / 2
    if kind in ("fixed", "pinned", "roller"):
        y = h - 6.0 if kind != "fixed" else h / 2
        x0 = 12.0 if kind == "fixed" else 6.0
        _line(d, (x0, y), (w - 6.0, y), C_MEMBER, 2.0)
        st = {"fixed": SupportType.FIXED, "pinned": SupportType.PINNED, "roller": SupportType.ROLLER}[kind]
        _draw_support(d, (x0 if kind == "fixed" else mid, y), st, (1.0, 0.0), u=6.0)
    elif kind == "point":
        _line(d, (6.0, 5.0), (w - 6.0, 5.0), C_MEMBER, 2.0)
        _arrow(d, (mid, h - 1.0), (mid, 6.2), C_LOAD, 1.4, 5.5)
    elif kind == "dist":
        _line(d, (6.0, 5.0), (w - 6.0, 5.0), C_MEMBER, 2.0)
        xs = np.linspace(8.0, w - 8.0, 5)
        for x in xs:
            _arrow(d, (float(x), h - 4.0), (float(x), 6.2), C_LOAD_Q, 0.7, 3.4)
        _line(d, (8.0, h - 4.0), (w - 8.0, h - 4.0), C_LOAD_Q, 0.9)
    elif kind == "moment":
        r = 8.0
        arc = [(mid + r * math.cos(t), h / 2 + r * math.sin(t)) for t in np.linspace(-0.8, 3.9, 28)]
        _poly(d, arc[:-2], C_LOAD, 1.3)
        _arrow(d, arc[-4], arc[-1], C_LOAD, 1.3, 4.5)
        d.add(Circle(mid, h / 2, 1.3, fillColor=C_MEMBER, strokeColor=None))
    elif kind == "global":
        _draw_triad(d, (mid - 10.0, 3.0), 14.0)
    elif kind == "local":
        a, b = (8.0, 4.0), (w - 8.0, h - 6.0)
        _line(d, a, b, C_MEMBER, 1.8)
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        c, s = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        o = (mid - s * 3.0, h / 2 - 1.0 + c * 3.0)
        _arrow(d, o, (o[0] + c * 12.0, o[1] + s * 12.0), C_AXIS, 0.8, 3.4)
        _arrow(d, o, (o[0] - s * 9.0, o[1] + c * 9.0), C_AXIS, 0.8, 3.4)
    elif kind == "deformed":
        _line(d, (6.0, h - 8.0), (w - 6.0, h - 8.0), C_GHOST, 1.2)
        xs = np.linspace(6.0, w - 6.0, 20)
        _poly(d, [(float(x), h - 8.0 - 10.0 * math.sin(math.pi * (x - 6.0) / (w - 12.0))) for x in xs],
              C_DEFORMED, 1.6)
    elif kind == "diagram":
        xs = np.linspace(6.0, w - 6.0, 20)
        pts = [(float(x), h - 6.0 - 14.0 * math.sin(math.pi * (x - 6.0) / (w - 12.0))) for x in xs]
        ring = [(6.0, h - 6.0), *pts, (w - 6.0, h - 6.0)]
        d.add(Polygon([cc for p in ring for cc in p], fillColor=_tint(C_DIAG["M"], 0.18), strokeColor=None))
        _poly(d, pts, C_DIAG["M"], 1.2)
        _line(d, (6.0, h - 6.0), (w - 6.0, h - 6.0), C_MEMBER, 1.4)
    elif kind == "dim":
        _draw_dim(d, (6.0, h / 2 - 3.0), (w - 6.0, h / 2 - 3.0), "2.50", True)
    return d


_SCRIPT_RE = re.compile(r"([_^])\{([^}]*)\}")
_GREEK_RE = re.compile("[σδνρθΣ]")


def _rl(text: str) -> str:
    """Mini-notación de core.glossary (x_{sub}, x^{sup}) -> markup de Paragraph de ReportLab."""
    out = _SCRIPT_RE.sub(lambda m: f"<sub>{m.group(2)}</sub>" if m.group(1) == "_" else
                         f"<super>{m.group(2)}</super>", escape(text))
    if not _UNICODE:  # sin DejaVu: griegas vía fuente Symbol estándar
        out = _GREEK_RE.sub(lambda m: f'<font face="Symbol">{m.group(0)}</font>', out)
    return out


def _section_glossary() -> list[Flowable]:
    """Sección 6. El contenido vive en core/glossary.py (fuente única con la wiki de la app)."""
    out: list[Flowable] = [_h1("6. Glosario y simbología"), _h2("6.1 Notación")]

    # Notación: tabla con filas de grupo
    data: list[list[Any]] = [[_p("Símbolo", "cell_h"), _p("Descripción", "cell_h"), _p("Unidad", "cell_hr")]]
    style: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]
    for group in NOTATION:
        r = len(data)
        data.append([_p(f"<b>{escape(group.title)}</b>", "cell"), "", ""])
        style += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), ZEBRA),
                  ("LINEABOVE", (0, r), (-1, r), 0.3, RULE)]
        for e in group.entries:
            data.append([_p(f"<b>{_rl(e.symbol)}</b>", "cell"), _p(_rl(e.description), "cell"),
                         _p(_rl(e.unit), "cell_r")])
    t = Table(data, colWidths=[0.16 * CONTENT_W, 0.70 * CONTENT_W, 0.14 * CONTENT_W], repeatRows=1,
              hAlign="LEFT")
    t.setStyle(TableStyle(style))
    out.append(t)

    # Símbolos gráficos: ícono vectorial + nombre + significado
    out.append(_h2("6.2 Símbolos gráficos"))
    gdata: list[list[Any]] = [[_p("Símbolo", "cell_h"), _p("Nombre", "cell_h"), _p("Significado", "cell_h")]]
    for gs in GRAPHIC_SYMBOLS:
        gdata.append([_icon(gs.kind), _p(f"<b>{_rl(gs.name)}</b>", "cell"), _p(_rl(gs.meaning), "cell")])
    g = Table(gdata, colWidths=[0.14 * CONTENT_W, 0.20 * CONTENT_W, 0.66 * CONTENT_W], repeatRows=1,
              hAlign="LEFT")
    gstyle: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 2.5),
    ]
    g.setStyle(TableStyle(gstyle))
    out.append(g)

    # Términos
    out.append(_h2("6.3 Términos"))
    rows = [[f"<b>{_rl(t.term)}</b>", _rl(t.definition)] for t in TERMS]
    out.append(_table(["Término", "Definición"], rows, [0.24, 0.76]))
    return out


# --------------------------------------------------------------------------- #
# Cálculos auxiliares de presentación
# --------------------------------------------------------------------------- #


_DIR_TXT = {LoadDirection.GLOBAL_Y: "Vertical (Y)", LoadDirection.GLOBAL_X: "Horizontal (X)",
            LoadDirection.LOCAL_Y: "Perpendicular a barra"}


@dataclass(frozen=True, slots=True)
class _LoadGroup:
    origin: str
    members: str
    p0: tuple[float, float]  # punto inicial (x, y) [mm]
    p1: tuple[float, float]  # punto final (x, y) [mm]
    direction: LoadDirection
    q0: float  # [N/mm]
    q1: float


def _group_distributed(model: Model) -> list[_LoadGroup]:
    """Une tramos contiguos de una misma carga (el builder la parte en barras) para listarla una vez."""
    groups: list[list[DistributedLoad]] = []
    for d in sorted(model.distributed_loads, key=lambda d: (d.label, d.direction.value, d.member)):
        if groups:
            p = groups[-1][-1]
            sp = (p.q_end - p.q_start) / model.member_length(p.member)
            sd = (d.q_end - d.q_start) / model.member_length(d.member)
            if (p.label == d.label and p.direction is d.direction
                    and model.members[p.member].j == model.members[d.member].i
                    and math.isclose(p.q_end, d.q_start, rel_tol=1e-6, abs_tol=1e-9)
                    and math.isclose(sp, sd, rel_tol=1e-6, abs_tol=1e-12)):
                groups[-1].append(d)
                continue
        groups.append([d])
    out = []
    for g in groups:
        first, last = model.members[g[0].member], model.members[g[-1].member]
        ids = f"{g[0].member}" if len(g) == 1 else f"{g[0].member}–{g[-1].member}"
        out.append(_LoadGroup(
            origin="Peso propio" if g[0].label == SELF_WEIGHT else (g[0].label or "Usuario"),
            members=ids, p0=model.nodes[first.i], p1=model.nodes[last.j],
            direction=g[0].direction, q0=g[0].q_start, q1=g[-1].q_end,
        ))
    return out


def _applied_resultant(model: Model) -> tuple[float, float]:
    """Resultante global (Fx, Fy) [N] de todas las cargas aplicadas."""
    Fx = sum(p.Fx for p in model.nodal_loads)
    Fy = sum(p.Fy for p in model.nodal_loads)
    for d in model.distributed_loads:
        Q = 0.5 * (d.q_start + d.q_end) * model.member_length(d.member)
        c, s = model.member_cos_sin(d.member)
        gx, gy = {LoadDirection.GLOBAL_X: (1.0, 0.0), LoadDirection.GLOBAL_Y: (0.0, 1.0),
                  LoadDirection.LOCAL_Y: (-s, c)}[d.direction]
        Fx += Q * gx
        Fy += Q * gy
    return Fx, Fy


def _horizontal_extent(model: Model) -> float:
    xs = [x for x, _ in model.nodes]
    return max(xs) - min(xs) or max(model.member_length(m) for m in range(len(model.members)))


# --------------------------------------------------------------------------- #
# Helpers de maquetación
# --------------------------------------------------------------------------- #


_GREEK = {"s": "σ", "d": "δ", "n": "ν", "r": "ρ"}


def _g(letter: str) -> str:
    """Letra griega en la fuente Symbol estándar (no requiere embeber TTF): 's' -> σ, 'd' -> δ."""
    return f'<font face="Symbol">{_GREEK[letter]}</font>'


def _sup(n: int) -> str:
    """Exponente con markup (Helvetica estándar no trae ⁴; se usa <super> para todos)."""
    return f"<super>{n}</super>"


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def _h1(text: str) -> Paragraph:
    return _p(text, "h1")


def _h2(text: str) -> Paragraph:
    return _p(text, "h2")


def _bullet(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ParagraphStyle(f"b_{style}", parent=STYLES[style], leftIndent=9,
                                          bulletIndent=0, spaceAfter=2, alignment=TA_LEFT),
                     bulletText="•")


def _num(v: float) -> str:
    v = 0.0 if abs(v) < 5e-10 else v  # evita "-0.000"
    return f"{v:,.3f}"


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _table(head: list[str], rows: list[list[str]], widths: list[float], num_cols: set[int] | None = None,
           total_row: bool = False) -> Table:
    num_cols = num_cols or set()
    data = [[_p(h, "cell_hr" if i in num_cols else "cell_h") for i, h in enumerate(head)]]
    data += [[_p(c, "cell_r" if i in num_cols else "cell") for i, c in enumerate(r)] for r in rows]
    t = Table(data, colWidths=[w * CONTENT_W for w in widths], repeatRows=1, hAlign="LEFT")
    style: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for r in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    if total_row:
        style.append(("LINEABOVE", (0, -1), (-1, -1), 0.5, RULE))
    t.setStyle(TableStyle(style))
    return t


def _kv_table(rows: list[list[str]]) -> Table:
    data = [[_p(k, "cell"), _p(v, "cell_r")] for k, v in rows]
    t = Table(data, colWidths=[0.62 * CONTENT_W, 0.38 * CONTENT_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t
