"""Memoria de cálculo en PDF (A4) con ReportLab.

Capa de presentación del core (ver AGENTS.md §3/§4): recibe objetos del core en
unidades SI-mm y los MUESTRA en unidades de ingeniería (m, kN, kN/m, kN·m, cm², cm⁴,
cm³). Las conversiones viven sólo en este módulo y nunca vuelven al resto del core.

Función pública:
    build_pdf_report(model, results, check, project_title, author, *, date, reference_length,
                     include_figures) -> bytes

Pura: no muta sus argumentos y, con `date` fijo, devuelve exactamente los mismos bytes
(PDF "invariant": sin marcas de tiempo ni ID aleatorio). Única lectura de disco: las fuentes
DejaVu Sans empaquetadas en `core/fonts/`, una vez al importar (AGENTS.md §3).

Las figuras (esquema, deformada, diagramas N-V-M-σ) se dibujan en vectores con
`reportlab.graphics` a partir de `Model`/`Results`: el core no depende de Plotly ni de un
navegador para exportar imágenes.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.sax.saxutils import escape

import numpy as np
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFError, TTFont
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
from core.model import SELF_WEIGHT, DistributedLoad, LoadDirection, Model, SupportType
from core.solver import MemberResult, Results
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

# --------------------------------------------------------------------------- #
# Tipografía: DejaVu Sans embebida (subset). Cubre σ δ Σ · ⁴ → y se ve igual en cualquier
# visor. Si faltaran los .ttf, cae a Helvetica + Symbol (estándar PDF, sin embeber).
# --------------------------------------------------------------------------- #

_FONT_DIR = Path(__file__).with_name("fonts")


def _register_fonts() -> tuple[str, str, bool]:
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(_FONT_DIR / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
    except (OSError, TTFError):
        return "Helvetica", "Helvetica-Bold", False
    # <b> en Paragraph -> DejaVuSans-Bold (sin itálica embebida: se usa la recta)
    addMapping("DejaVuSans", 0, 0, "DejaVuSans")
    addMapping("DejaVuSans", 1, 0, "DejaVuSans-Bold")
    addMapping("DejaVuSans", 0, 1, "DejaVuSans")
    addMapping("DejaVuSans", 1, 1, "DejaVuSans-Bold")
    return "DejaVuSans", "DejaVuSans-Bold", True


_FONT, _BOLD, _UNICODE = _register_fonts()

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
    "caption": ParagraphStyle("caption", fontName=_FONT, fontSize=7.8, leading=10, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=1, spaceAfter=4),
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
    include_figures: bool = True,
    include_glossary: bool = True,
) -> bytes:
    """Genera la memoria de cálculo y devuelve el PDF como bytes (empieza con b"%PDF").

    Args:
        model, results, check: salida de `solve()` y `check_safety()` para el mismo modelo.
        project_title, author: texto libre del encabezado.
        date: fecha del cálculo (por defecto, hoy). Fijarla hace la salida reproducible.
        reference_length: longitud de referencia para la flecha relativa L/δ [mm]
            (por defecto, la extensión horizontal de la estructura).
        include_figures: agrega esquema de cargas, deformada y diagramas N-V-M-σ (vectoriales).
        include_glossary: agrega la sección final "Glosario y simbología".
    """
    date = date or dt.date.today()
    title = project_title.strip() or "Memoria de Cálculo"
    author = author.strip()
    L_ref = reference_length if reference_length else _horizontal_extent(model)

    buf = io.BytesIO()
    doc = _Doc(buf, title=title, author=author or APP_NAME, date=date)
    figs = _Figures() if include_figures else None
    story: list[Flowable] = []
    story += _title_block(model, check, title, author, date)
    story += _section_basis(check, figs, include_glossary)
    story += _section_inputs(model, figs)
    story += _section_results(model, results, L_ref, figs)
    story += _section_verification(check)
    story += _section_limitations()
    if include_glossary:
        story += _section_glossary()
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
        canv.drawString(MARGIN_X, top, APP_NAME.upper())
        if doc.page > 1:  # en la pág. 1 el título ya está en grande: no repetirlo
            canv.setFont(_FONT, 7.5)
            canv.setFillColor(MUTED)
            canv.drawRightString(PAGE_W - MARGIN_X, top, _clip(self.report_title, 80))
        canv.setStrokeColor(ACCENT)
        canv.setLineWidth(0.8)
        canv.line(MARGIN_X, top - 2.5 * mm, PAGE_W - MARGIN_X, top - 2.5 * mm)
        bottom = 11 * mm
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN_X, bottom + 4 * mm, PAGE_W - MARGIN_X, bottom + 4 * mm)
        canv.drawString(MARGIN_X, bottom,
                        f"Cálculo elástico lineal 2D · Euler-Bernoulli · {self.report_date:%d/%m/%Y}")
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


def _title_block(model: Model, check: SafetyCheck, title: str, author: str, date: dt.date
                 ) -> list[Flowable]:
    kind = "Viga recta" if model.is_horizontal_beam() else "Pórtico plano"
    meta = [
        ("AUTOR", escape(author) or "—"),
        ("FECHA", f"{date:%d/%m/%Y}"),
        ("ESTRUCTURA", f"{kind} · {len(model.nodes)} nodos · {len(model.members)} barras"),
        ("MÉTODO", "Rigidez directa · Euler-Bernoulli"),
        ("CRITERIO", f"{_g('s')} elástica · FS ≥ {check.fs_min:g}"),
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
    # El subtítulo nombra el tipo de documento sólo si el título no lo hace ya
    is_memo = "memoria de c" in title.casefold()
    subtitle = ("Predimensionamiento estructural 2D · análisis elástico lineal" if is_memo
                else "Memoria de cálculo · predimensionamiento estructural 2D")
    return [
        _p(escape(title), "title"),
        _p(subtitle, "subtitle"),
        Spacer(1, 5 * mm),
        t,
        Spacer(1, 2 * mm),
    ]


def _section_basis(check: SafetyCheck, figs: _Figures | None, glossary: bool = False) -> list[Flowable]:
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
    if glossary:
        items.append("La notación, los símbolos gráficos y los términos usados se detallan en la "
                     "sección 6 (Glosario y simbología).")
    out: list[Flowable] = [_h1("1. Bases de cálculo"), *(_bullet(t) for t in items)]
    if figs is not None:
        out.append(figs.add(_fig_conventions(CONTENT_W),
                            "Convenciones de signos usadas en toda la memoria: ejes globales, "
                            "solicitaciones positivas sobre un elemento dx y ejes locales de barra."))
    return out


def _section_inputs(model: Model, figs: _Figures | None) -> list[Flowable]:
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
                      rows, [0.24, 0.10, 0.12, 0.14, 0.12, 0.14, 0.14], num_cols={1, 2, 3, 4, 5, 6}))

    # 2.5 Esquema
    if figs is not None:
        out.append(figs.add(_fig_structure(model, CONTENT_W),
                            "Esquema estático: apoyos, cargas de usuario, numeración de nodos, ejes "
                            "globales" + ("" if model.is_horizontal_beam() else " y locales x'-y' por barra")
                            + ". Cotas en m.", heading=_h2("2.5 Esquema estático y cargas")))
    return out


def _section_results(model: Model, results: Results, L_ref: float, figs: _Figures | None
                     ) -> list[Flowable]:
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
    if figs is not None:
        out += _diagram_figures(model, results, figs)

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
    if figs is not None:
        drawing, k = _fig_deformed(model, results, CONTENT_W)
        out.append(figs.add(drawing, f"Deformada amplificada ×{k:g} (en gris, la geometría sin deformar). "
                                     f"{_g('d')}<sub>máx</sub> = {abs(d.value):,.2f} mm."))
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


def _chain_loads(model: Model, loads: Sequence[DistributedLoad]) -> list[list[DistributedLoad]]:
    """Une tramos contiguos de una misma carga (el builder la parte en barras)."""
    groups: list[list[DistributedLoad]] = []
    for d in sorted(loads, key=lambda d: (d.label, d.direction.value, d.member)):
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
    return groups


def _group_distributed(model: Model) -> list[_LoadGroup]:
    """Cargas distribuidas agrupadas para listarlas una vez por tramo continuo."""
    out = []
    for g in _chain_loads(model, model.distributed_loads):
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
# Figuras vectoriales (reportlab.graphics). Coordenadas de dibujo en puntos (pt),
# origen abajo-izquierda, Y hacia arriba: misma orientación que los ejes globales.
# --------------------------------------------------------------------------- #

C_MEMBER = colors.HexColor("#2B3440")
C_GHOST = colors.HexColor("#B8C0CA")
C_SUPPORT = colors.HexColor("#5B6573")
C_LOAD = colors.HexColor("#C2410C")
C_LOAD_Q = colors.HexColor("#EA580C")
C_DEFORMED = colors.HexColor("#2563EB")
C_DIM = colors.HexColor("#7B8794")
C_AXIS = colors.HexColor("#334E68")  # ejes de referencia (globales y locales)
C_DIAG = {"M": colors.HexColor("#7C3AED"), "V": colors.HexColor("#0891B2"),
          "N": colors.HexColor("#16A34A"), "sigma": colors.HexColor("#DC2626")}

# magnitud -> (título, unidad, factor de impresión)
_DIAG_SPEC: dict[str, tuple[str, str, float]] = {
    "M": ("Momento flector M", "kN·m", _KNM),
    "V": ("Esfuerzo de corte V", "kN", _KN),
    "N": ("Esfuerzo normal N", "kN", _KN),
    "sigma": ("Tensión normal máxima |σ|", "MPa", 1.0),
}

Pt = tuple[float, float]


class _Figures:
    """Numera las figuras y arma dibujo + epígrafe."""

    def __init__(self) -> None:
        self.n = 0

    def caption(self, text: str) -> Paragraph:
        self.n += 1
        return _p(f"<b>Figura {self.n}.</b> {text}", "caption")

    def add(self, drawing: Drawing, text: str, heading: Flowable | None = None) -> Flowable:
        head = [heading] if heading is not None else []
        return KeepTogether([*head, Spacer(1, 1.5 * mm), drawing, self.caption(text)])


@dataclass(frozen=True, slots=True)
class _View:
    """Transformación mundo [mm] -> dibujo [pt] con escala uniforme."""

    s: float
    ox: float
    oy: float

    def __call__(self, x: float, y: float) -> Pt:
        return x * self.s + self.ox, y * self.s + self.oy


def _fit(pts: Sequence[Pt], w: float, h: float, pad: tuple[float, float, float, float]) -> _View:
    """Encaja `pts` en w×h dejando pad = (izq, der, abajo, arriba) [pt], centrado."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    iw, ih = w - pad[0] - pad[1], h - pad[2] - pad[3]
    s = min(iw / (x1 - x0) if x1 > x0 else math.inf, ih / (y1 - y0) if y1 > y0 else math.inf)
    s = 1.0 if math.isinf(s) else s
    return _View(s, pad[0] + (iw - (x1 - x0) * s) / 2 - x0 * s, pad[2] + (ih - (y1 - y0) * s) / 2 - y0 * s)


def _auto_height(model: Model, w: float, pad: tuple[float, float, float, float],
                 h_min: float, h_max: float, extra: Sequence[Pt] = ()) -> float:
    """Alto del dibujo que respeta la proporción de la geometría, acotado a [h_min, h_max]."""
    pts = list(model.nodes) + list(extra)
    dx = max(p[0] for p in pts) - min(p[0] for p in pts)
    dy = max(p[1] for p in pts) - min(p[1] for p in pts)
    iw = w - pad[0] - pad[1]
    ih = iw * dy / dx if dx > 0 else h_max
    return float(min(max(ih + pad[2] + pad[3], h_min), h_max))


def _tint(c: colors.Color, a: float) -> colors.Color:
    """Color aclarado hacia blanco (relleno opaco: evita transparencias en el PDF)."""
    return colors.Color(1 - a * (1 - c.red), 1 - a * (1 - c.green), 1 - a * (1 - c.blue))


def _line(d: Drawing, a: Pt, b: Pt, color: colors.Color, width: float = 1.0,
          dash: Sequence[float] = ()) -> None:
    d.add(Line(a[0], a[1], b[0], b[1], strokeColor=color, strokeWidth=width,
               strokeDashArray=dash or None, strokeLineCap=1))  # type: ignore[arg-type]


def _poly(d: Drawing, pts: Sequence[Pt], color: colors.Color, width: float = 1.0) -> None:
    d.add(PolyLine([c for p in pts for c in p], strokeColor=color, strokeWidth=width,
                   strokeLineJoin=1, strokeLineCap=1))


def _arrow(d: Drawing, tail: Pt, head: Pt, color: colors.Color, width: float = 1.0,
           head_len: float = 5.0) -> None:
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    hl = min(head_len, 0.6 * L)
    hw = 0.42 * hl
    bx, by = head[0] - ux * hl, head[1] - uy * hl
    _line(d, tail, (bx, by), color, width)
    d.add(Polygon([head[0], head[1], bx - uy * hw, by + ux * hw, bx + uy * hw, by - ux * hw],
                  fillColor=color, strokeColor=color, strokeWidth=0.3))


_Anchor = Literal["start", "middle", "end"]


def _label(d: Drawing, x: float, y: float, text: str, size: float = 7.0, color: colors.Color = INK,
           anchor: _Anchor = "middle", bold: bool = False, halo: bool = True) -> None:
    """Texto con fondo blanco para que no lo tapen líneas. (x, y) = línea base."""
    font = _BOLD if bold else _FONT
    if halo:
        tw = stringWidth(text, font, size)
        lx = {"start": x, "middle": x - tw / 2, "end": x - tw}[anchor]
        d.add(Rect(lx - 1.2, y - 0.22 * size - 0.6, tw + 2.4, 0.95 * size + 1.2,
                   fillColor=colors.white, strokeColor=None))  # type: ignore[arg-type]  # stub incompleto
    d.add(String(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def _short(v: float) -> str:
    """Número compacto para rótulos de figura."""
    a = abs(v)
    if a < 5e-10:
        return "0"
    return f"{v:,.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def _member_dir(model: Model, node: int) -> Pt:
    """Dirección unitaria de la primera barra conectada, saliendo del nodo."""
    for m, mem in enumerate(model.members):
        if node in (mem.i, mem.j):
            c, s = model.member_cos_sin(m)
            return (c, s) if mem.i == node else (-c, -s)
    return 1.0, 0.0


def _draw_members(d: Drawing, model: Model, view: _View, color: colors.Color = C_MEMBER,
                  width: float = 2.2) -> None:
    for mem in model.members:
        _line(d, view(*model.nodes[mem.i]), view(*model.nodes[mem.j]), color, width)


def _draw_support(d: Drawing, p: Pt, kind: SupportType, direction: Pt, u: float = 7.5) -> None:
    if kind is SupportType.FIXED:
        dx, dy = direction
        px, py = -dy, dx  # muro perpendicular a la barra, del lado opuesto a ella
        _line(d, (p[0] + px * u, p[1] + py * u), (p[0] - px * u, p[1] - py * u), C_SUPPORT, 1.8)
        for t in np.linspace(-1.0, 1.0, 6):
            b = (p[0] + px * u * float(t), p[1] + py * u * float(t))
            _line(d, b, (b[0] - dx * 0.55 * u - px * 0.35 * u, b[1] - dy * 0.55 * u - py * 0.35 * u),
                  C_SUPPORT, 0.6)
        return
    base = p[1] - 1.1 * u
    d.add(Polygon([p[0], p[1], p[0] - 0.7 * u, base, p[0] + 0.7 * u, base],
                  fillColor=_tint(C_SUPPORT, 0.18), strokeColor=C_SUPPORT, strokeWidth=0.9))
    if kind is SupportType.ROLLER:
        for sx in (-0.35, 0.35):
            d.add(Circle(p[0] + sx * u, base - 1.9, 1.6, fillColor=colors.white,
                         strokeColor=C_SUPPORT, strokeWidth=0.7))
        base -= 3.8
    _line(d, (p[0] - u, base), (p[0] + u, base), C_SUPPORT, 1.0)
    for t in np.linspace(-0.9, 0.9, 6):
        x = p[0] + float(t) * u
        _line(d, (x, base), (x - 0.3 * u, base - 0.3 * u), C_SUPPORT, 0.6)


def _draw_supports(d: Drawing, model: Model, view: _View) -> None:
    for sup in model.supports:
        _draw_support(d, view(*model.nodes[sup.node]), sup.type, _member_dir(model, sup.node))


def _load_dir(model: Model, dl: DistributedLoad) -> Pt:
    """Dirección unitaria global de la carga para q > 0."""
    c, s = model.member_cos_sin(dl.member)
    return {LoadDirection.GLOBAL_Y: (0.0, 1.0), LoadDirection.GLOBAL_X: (1.0, 0.0),
            LoadDirection.LOCAL_Y: (-s, c)}[dl.direction]


def _user_dist_loads(model: Model) -> list[DistributedLoad]:
    """Cargas distribuidas de usuario sumadas por (barra, dirección); sin peso propio."""
    acc: dict[tuple[int, LoadDirection], list[float]] = {}
    for dl in model.distributed_loads:
        if dl.label == SELF_WEIGHT:
            continue
        a = acc.setdefault((dl.member, dl.direction), [0.0, 0.0])
        a[0] += dl.q_start
        a[1] += dl.q_end
    return [DistributedLoad(m, q1, q2, direction) for (m, direction), (q1, q2) in acc.items() if q1 or q2]


def _draw_dist_loads(d: Drawing, model: Model, view: _View, H: float = 20.0) -> None:
    loads = _user_dist_loads(model)
    if not loads:
        return
    qmax = max(max(abs(dl.q_start), abs(dl.q_end)) for dl in loads)
    busy = [view(*model.nodes[p.node]) for p in model.nodal_loads]  # zonas con rótulo de carga puntual
    for chain in _chain_loads(model, loads):
        outline: list[Pt] = []
        for dl in chain:
            mem = model.members[dl.member]
            a, b = view(*model.nodes[mem.i]), view(*model.nodes[mem.j])
            gx, gy = _load_dir(model, dl)
            n = max(2, round(math.hypot(b[0] - a[0], b[1] - a[1]) / 24.0) + 1)
            for k in range(n):
                t = k / (n - 1)
                q = dl.q_start + (dl.q_end - dl.q_start) * t
                head = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                h = H * q / qmax
                tail = (head[0] - gx * h, head[1] - gy * h)
                outline.append(tail)
                if abs(q) > 0.03 * qmax:
                    gap = math.copysign(1.2, q)  # la punta no tapa la barra
                    _arrow(d, tail, (head[0] - gx * gap, head[1] - gy * gap), C_LOAD_Q, 0.7, 3.6)
        _poly(d, outline, C_LOAD_Q, 0.9)
        q1, q2 = chain[0].q_start, chain[-1].q_end
        uniform = all(math.isclose(c.q_start, c.q_end) for c in chain) and math.isclose(q1, q2)
        text = f"q = {abs(q1):.3g} kN/m" if uniform else f"q = {abs(q1):.3g} {_TO} {abs(q2):.3g} kN/m"
        gx, gy = _load_dir(model, chain[0])
        sgn = 1.0 if (q1 + q2) >= 0 else -1.0
        # Rótulo en el punto del contorno más alejado de las cargas puntuales (evita superposición)
        n_o = len(outline)
        cands = [outline[min(n_o - 1, round(f * (n_o - 1)))] for f in (0.5, 0.3, 0.7, 0.15, 0.85)]
        mx, my = next((c for c in cands if all(abs(c[0] - b[0]) > 45.0 for b in busy)), cands[0])
        ox, oy = -gx * sgn * 6.0, -gy * sgn * 6.0
        if abs(gy) >= abs(gx):
            _label(d, mx, my + oy + (0.0 if oy > 0 else -7.0), text, 7.2, C_LOAD)
        else:
            _label(d, mx + ox, my - 2.5, text, 7.2, C_LOAD, anchor="end" if ox < 0 else "start")


def _draw_nodal_loads(d: Drawing, model: Model, view: _View, L: float = 26.0) -> None:
    for p in model.nodal_loads:
        x, y = view(*model.nodes[p.node])
        F = math.hypot(p.Fx, p.Fy)
        if F > 0:
            ux, uy = p.Fx / F, p.Fy / F
            tail = (x - ux * L, y - uy * L)
            _arrow(d, tail, (x - ux * 1.2, y - uy * 1.2), C_LOAD, 1.5, 6.5)
            text = f"{_short(F * _KN)} kN"
            if abs(uy) >= abs(ux):
                _label(d, tail[0], tail[1] + (3.0 if uy < 0 else -9.0), text, 7.2, C_LOAD)
            else:
                _label(d, tail[0] - math.copysign(3.0, ux), tail[1] - 2.5, text, 7.2, C_LOAD,
                       anchor="end" if ux > 0 else "start")
        if p.Mz:
            r = 10.0
            a0, a1 = (-0.25 * math.pi, 1.25 * math.pi) if p.Mz > 0 else (1.25 * math.pi, -0.25 * math.pi)
            arc = [(x + r * math.cos(t), y + r * math.sin(t)) for t in np.linspace(a0, a1, 32)]
            _poly(d, arc[:-2], C_LOAD, 1.3)
            _arrow(d, arc[-4], arc[-1], C_LOAD, 1.3, 5.0)
            _label(d, x - 3.0, y - r - 9.0, f"{_short(abs(p.Mz) * _KNM)} kN·m", 7.2, C_LOAD, anchor="end")


def _draw_node_numbers(d: Drawing, model: Model, view: _View) -> None:
    for k, (x, y) in enumerate(model.nodes):
        px, py = view(x, y)
        _label(d, px + 7.0, py - 11.0, str(k), 6.2, MUTED, anchor="start", halo=False)


def _draw_dim(d: Drawing, a: Pt, b: Pt, text: str, horizontal: bool) -> None:
    """Cota entre a y b (ya ubicados en la línea de cota)."""
    _line(d, a, b, C_DIM, 0.6)
    t = 3.0
    for p in (a, b):
        if horizontal:
            _line(d, (p[0], p[1] - t), (p[0], p[1] + t), C_DIM, 0.6)
        else:
            _line(d, (p[0] - t, p[1]), (p[0] + t, p[1]), C_DIM, 0.6)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    if horizontal:
        _label(d, mx, my + 2.5, text, 6.8, C_DIM)
    else:
        _label(d, mx + 4.0, my - 2.5, text, 6.8, C_DIM, anchor="start")


def _draw_dimensions(d: Drawing, model: Model, view: _View, below: float) -> None:
    xs = sorted({round(x, 6) for x, _ in model.nodes})
    y_min = min(y for _, y in model.nodes)
    yd = view(0.0, y_min)[1] - below
    if model.is_horizontal_beam():
        last_label = -1e9
        for xa, xb in zip(xs, xs[1:], strict=False):
            pa, pb = view(xa, y_min), view(xb, y_min)
            text = f"{(xb - xa) * _M:.2f}"
            fits = (pb[0] - pa[0]) > stringWidth(text, _FONT, 6.8) + 4 and pa[0] > last_label
            _draw_dim(d, (pa[0], yd), (pb[0], yd), text if fits else "", True)
            if fits:
                last_label = (pa[0] + pb[0]) / 2
        if len(xs) > 2:
            pa, pb = view(xs[0], y_min), view(xs[-1], y_min)
            _draw_dim(d, (pa[0], yd - 13), (pb[0], yd - 13), f"L = {(xs[-1] - xs[0]) * _M:.2f} m", True)
        return
    pa, pb = view(xs[0], y_min), view(xs[-1], y_min)
    _draw_dim(d, (pa[0], yd), (pb[0], yd), f"{(xs[-1] - xs[0]) * _M:.2f} m", True)
    y_max = max(y for _, y in model.nodes)
    x_r = view(xs[-1], 0.0)[0] + 22.0
    _draw_dim(d, (x_r, view(0.0, y_min)[1]), (x_r, view(0.0, y_max)[1]), f"{(y_max - y_min) * _M:.2f} m",
              False)


def _draw_triad(d: Drawing, o: Pt, size: float = 15.0, moment: bool = True) -> None:
    """Terna de ejes globales X-Y con origen en `o` y sentido positivo de Mz (antihorario)."""
    _arrow(d, o, (o[0] + size, o[1]), C_AXIS, 0.9, 4.2)
    _arrow(d, o, (o[0], o[1] + size), C_AXIS, 0.9, 4.2)
    _label(d, o[0] + size + 2.0, o[1] - 2.4, "X", 6.8, C_AXIS, anchor="start", bold=True, halo=False)
    _label(d, o[0], o[1] + size + 2.2, "Y", 6.8, C_AXIS, bold=True, halo=False)
    d.add(Circle(o[0], o[1], 1.1, fillColor=C_AXIS, strokeColor=None))
    if moment:
        r = 0.55 * size
        arc = [(o[0] + r * math.cos(t), o[1] + r * math.sin(t)) for t in np.linspace(0.18, 1.4, 14)]
        _poly(d, arc[:-1], C_AXIS, 0.7)
        _arrow(d, arc[-3], arc[-1], C_AXIS, 0.7, 3.2)
        _label(d, o[0] + 0.62 * size, o[1] + 0.62 * size, "+Mz", 5.8, C_AXIS, anchor="start", halo=False)


def _draw_local_axes(d: Drawing, model: Model, view: _View, t: float = 0.28, L: float = 13.0,
                     with_y: bool = True) -> None:
    """Ejes locales de cada barra: x' de i a j; y' = x' girado +90°. Signos de N y V referidos a ellos."""
    for m, mem in enumerate(model.members):
        c, s = model.member_cos_sin(m)
        a, b = view(*model.nodes[mem.i]), view(*model.nodes[mem.j])
        o = (a[0] + (b[0] - a[0]) * t - s * 5.0, a[1] + (b[1] - a[1]) * t + c * 5.0)  # corrido hacia +y'
        _arrow(d, o, (o[0] + c * L, o[1] + s * L), C_AXIS, 0.8, 3.6)
        _label(d, o[0] + c * (L + 5.0), o[1] + s * (L + 5.0) - 2.3, "x'", 6.2, C_AXIS, bold=True)
        if with_y:
            _arrow(d, o, (o[0] - s * 0.75 * L, o[1] + c * 0.75 * L), C_AXIS, 0.8, 3.6)
            _label(d, o[0] - s * (0.75 * L + 5.0), o[1] + c * (0.75 * L + 5.0) - 2.3, "y'", 6.2, C_AXIS,
                   bold=True)


def _fig_conventions(w: float, h: float = 92.0) -> Drawing:
    """Convenciones de signos: ejes globales, N-V-M positivos sobre un elemento dx, ejes locales."""
    d = Drawing(w, h)
    cw = w / 5.0
    cy = 48.0
    ew, eh = 30.0, 16.0  # elemento diferencial

    def element(cx: float) -> None:
        style: dict[str, Any] = {"fillColor": _tint(C_MEMBER, 0.10), "strokeColor": C_MEMBER,
                                 "strokeWidth": 0.8}
        d.add(Rect(cx - ew / 2, cy - eh / 2, ew, eh, **style))
        _label(d, cx, cy - 2.4, "dx", 6.0, MUTED, halo=False)

    def caption(cx: float, line1: str, line2: str = "") -> None:
        _label(d, cx, 14.0, line1, 6.8, INK, halo=False)
        if line2:
            _label(d, cx, 5.5, line2, 6.2, MUTED, halo=False)

    # 1) Ejes globales
    cx = cw * 0.5
    _draw_triad(d, (cx - 12.0, cy - 14.0), 26.0)
    caption(cx, "Ejes globales", "Mz y reacciones: + antihorario")
    # 2) N > 0: tracción (flechas salientes)
    cx = cw * 1.5
    element(cx)
    for sgn in (-1.0, 1.0):
        face = cx + sgn * ew / 2
        _arrow(d, (face, cy), (face + sgn * 15.0, cy), C_DIAG["N"], 1.2, 4.5)
        _label(d, face + sgn * 9.0, cy + 4.0, "N", 6.8, C_DIAG["N"], bold=True, halo=False)
    caption(cx, "N > 0: tracción")
    # 3) V > 0: horario (cara izq. hacia arriba, cara der. hacia abajo); V = dM/dx
    cx = cw * 2.5
    element(cx)
    for sgn in (-1.0, 1.0):
        xf = cx + sgn * (ew / 2 + 4.0)
        up = sgn < 0  # cara izquierda: hacia arriba; cara derecha: hacia abajo
        _arrow(d, (xf, cy - 13.0 if up else cy + 13.0), (xf, cy + 13.0 if up else cy - 13.0),
               C_DIAG["V"], 1.2, 4.5)
        _label(d, xf + sgn * 6.0, cy - 2.4, "V", 6.8, C_DIAG["V"], bold=True, halo=False)
    caption(cx, "V > 0", "V = dM/dx")
    # 4) M > 0: tracción inferior (cara izq. horario, cara der. antihorario)
    cx = cw * 3.5
    element(cx)
    for sgn in (-1.0, 1.0):
        fx = cx + sgn * ew / 2
        ts = np.linspace(1.5 * math.pi, 0.5 * math.pi, 16) if sgn < 0 else np.linspace(-0.5 * math.pi,
                                                                                      0.5 * math.pi, 16)
        r = 10.0
        arc = [(fx + sgn * 2.0 + r * math.cos(float(t)), cy + r * math.sin(float(t))) for t in ts]
        arc = [p for p in arc if (p[0] - fx) * sgn >= -0.5]  # sólo el lado exterior
        _poly(d, arc[:-1], C_DIAG["M"], 1.1)
        _arrow(d, arc[-3], arc[-1], C_DIAG["M"], 1.1, 4.0)
        _label(d, fx + sgn * 16.0, cy - 2.4, "M", 6.8, C_DIAG["M"], bold=True, halo=False)
    caption(cx, "M > 0: tracción inferior", "(y' negativa)")
    # 5) Ejes locales de una barra genérica
    cx = cw * 4.5
    i, j = (cx - 26.0, cy - 16.0), (cx + 22.0, cy + 14.0)
    _line(d, i, j, C_MEMBER, 2.0)
    for p, name, dx in ((i, "i", -6.0), (j, "j", 6.0)):
        d.add(Circle(p[0], p[1], 1.8, fillColor=colors.white, strokeColor=C_MEMBER, strokeWidth=0.8))
        _label(d, p[0] + dx, p[1] - 2.4, name, 6.8, C_MEMBER, bold=True, halo=False)
    L = math.hypot(j[0] - i[0], j[1] - i[1])
    c, s = (j[0] - i[0]) / L, (j[1] - i[1]) / L
    o = ((i[0] + j[0]) / 2 - s * 4.0, (i[1] + j[1]) / 2 + c * 4.0)
    _arrow(d, o, (o[0] + c * 16.0, o[1] + s * 16.0), C_AXIS, 0.9, 4.0)
    _arrow(d, o, (o[0] - s * 12.0, o[1] + c * 12.0), C_AXIS, 0.9, 4.0)
    _label(d, o[0] + c * 21.0, o[1] + s * 21.0 - 2.3, "x'", 6.8, C_AXIS, bold=True, halo=False)
    _label(d, o[0] - s * 17.0, o[1] + c * 17.0 - 2.3, "y'", 6.8, C_AXIS, bold=True, halo=False)
    caption(cx, "Ejes locales de barra", "x' de i a j · y' a +90°")
    return d


def _fig_structure(model: Model, w: float) -> Drawing:
    beam = model.is_horizontal_beam()
    pad = (34.0, 34.0, 62.0, 44.0) if beam else (64.0, 70.0, 56.0, 46.0)
    h = 140.0 if beam else _auto_height(model, w, pad, 200.0, 290.0)
    d = Drawing(w, h)
    view = _fit(model.nodes, w, h, pad)
    _draw_members(d, model, view)
    _draw_supports(d, model, view)
    _draw_dist_loads(d, model, view)
    _draw_nodal_loads(d, model, view)
    _draw_node_numbers(d, model, view)
    _draw_dimensions(d, model, view, below=34.0 if beam else 30.0)
    _draw_triad(d, (8.0, h - 26.0))
    if not beam:  # en vigas horizontales los locales coinciden con los globales
        _draw_local_axes(d, model, view)
    if any(dl.label == SELF_WEIGHT for dl in model.distributed_loads):
        _label(d, 2.0, 3.0, "+ peso propio (incluido en el cálculo, no dibujado)", 6.5, MUTED,
               anchor="start", halo=False)
    return d


def _nice(v: float) -> float:
    """Redondea hacia abajo a 1-2-5 × 10ⁿ (escala legible)."""
    if v <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(v))
    m = v / e
    return (5.0 if m >= 5 else 2.0 if m >= 2 else 1.0) * e


def _fig_deformed(model: Model, results: Results, w: float) -> tuple[Drawing, float]:
    """Deformada amplificada; devuelve (dibujo, factor de amplificación)."""
    xs = [x for x, _ in model.nodes]
    ys = [y for _, y in model.nodes]
    ext = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    dmax = max(float(mr.deflection.max()) for mr in results.members)
    k = _nice(0.10 * ext / dmax) if dmax > 0 else 1.0
    deformed = [mr.points + k * mr.displacement for mr in results.members]
    pts: list[Pt] = list(model.nodes) + [(float(p[0]), float(p[1])) for arr in deformed for p in arr]
    beam = model.is_horizontal_beam()
    pad = (46.0, 24.0, 22.0, 18.0)
    h = _auto_height(model, w, pad, 70.0, 150.0, extra=pts) if beam else \
        _auto_height(model, w, pad, 190.0, 280.0, extra=pts)
    d = Drawing(w, h)
    view = _fit(pts, w, h, pad)
    _draw_members(d, model, view, C_GHOST, 1.2)
    _draw_supports(d, model, view)
    _draw_triad(d, (8.0, 8.0), 13.0, moment=False)
    for arr in deformed:
        _poly(d, [view(float(p[0]), float(p[1])) for p in arr], C_DEFORMED, 1.8)
    e = results.extreme("deflection")
    mr = results.members[e.member]
    idx = int(np.argmin(np.abs(mr.x - e.x)))
    px, py = view(*map(float, mr.points[idx] + k * mr.displacement[idx]))
    d.add(Circle(px, py, 2.4, fillColor=C_DEFORMED, strokeColor=colors.white, strokeWidth=0.6))
    below = mr.displacement[idx][1] < 0
    text = f"{_gt('d')}máx = {abs(e.value):.2f} mm"
    half = stringWidth(text, _BOLD, 7.2) / 2 + 2.0
    _label(d, min(max(px, half), w - half), py + (-11.0 if below else 5.0), text, 7.2, C_DEFORMED,
           bold=True)  # acotado al ancho del dibujo
    return d, k


def _values(mr: MemberResult, q: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    return {"M": mr.M, "V": mr.V, "N": mr.N, "sigma": mr.sigma_abs}[q]


def _fig_beam_diagram(model: Model, results: Results, q: str, w: float, h: float = 96.0) -> Drawing:
    """Diagrama cartesiano f(x) para vigas. M se dibuja del lado traccionado (+ hacia abajo)."""
    f = _DIAG_SPEC[q][2]  # unidad y convención van en el epígrafe
    color = C_DIAG[q]
    x0 = min(x for x, _ in model.nodes)
    xs = np.concatenate([mr.points[:, 0] for mr in results.members]) - x0
    vs = np.concatenate([_values(mr, q) for mr in results.members]) * f
    plot = -vs if q == "M" else vs
    L = float(xs.max())
    pl, pr, pb, pt = 50.0, 24.0, 22.0, 16.0
    lo, hi = min(float(plot.min()), 0.0), max(float(plot.max()), 0.0)
    span = hi - lo if hi - lo > 1e-12 else 1.0
    sx, sy = (w - pl - pr) / L, (h - pb - pt) / span

    def P(x: float, v: float) -> Pt:
        return pl + x * sx, pb + (v - lo) * sy

    d = Drawing(w, h)
    for sup in model.supports:  # guías de apoyos
        xa = P(model.nodes[sup.node][0] - x0, 0.0)[0]
        _line(d, (xa, pb - 4), (xa, h - pt + 4), C_GHOST, 0.5, [1.5, 2])
    poly = [P(0.0, 0.0)] + [P(float(x), float(v)) for x, v in zip(xs, plot, strict=True)] + [P(L, 0.0)]
    d.add(Polygon([c for p in poly for c in p], fillColor=_tint(color, 0.16), strokeColor=None))
    _poly(d, poly[1:-1], color, 1.5)
    _line(d, P(0.0, 0.0), P(L, 0.0), C_MEMBER, 1.4)
    # Eje x (sentido +x) y eje de la magnitud con flecha hacia su sentido positivo
    z = P(L, 0.0)
    _arrow(d, z, (z[0] + 14.0, z[1]), C_AXIS, 0.8, 4.0)
    _label(d, z[0] + 16.0, z[1] - 2.4, "x", 6.8, C_AXIS, anchor="start", bold=True, halo=False)
    xa, y_bot, y_top = pl - 16.0, pb - 4.0, h - pt + 4.0
    pos_down = q == "M"  # M positivo (tracción inferior) se dibuja hacia abajo
    tail, head = ((xa, y_top), (xa, y_bot)) if pos_down else ((xa, y_bot), (xa, y_top))
    _arrow(d, tail, head, C_AXIS, 0.8, 4.0)
    sym = {"M": "+M", "V": "+V", "N": "+N", "sigma": f"|{_gt('s')}|"}[q]
    _label(d, xa, head[1] + (-9.0 if pos_down else 3.0), sym, 6.8, C_AXIS, bold=True, halo=False)
    for val in sorted({0.0, float(vs.max()), float(vs.min())}):  # marcas con el valor real (con signo)
        py = P(0.0, -val if pos_down else val)[1]
        _line(d, (xa - 2.5, py), (xa + 2.5, py), C_AXIS, 0.7)
        _label(d, xa - 4.0, py - 2.3, _short(val), 6.0, C_AXIS, anchor="end", halo=False)
    # Eje x: posiciones de nodos [m]
    last = -1e9
    for x in sorted({round(float(x), 6) for x, _ in ((nx - x0, 0) for nx, _ in model.nodes)}):
        px = P(x, 0.0)[0]
        _line(d, (px, pb - 9), (px, pb - 6), C_DIM, 0.5)
        if px - last > 24:
            _label(d, px, pb - 17, f"{x * _M:.2f}", 6.3, C_DIM, halo=False)
            last = px
    # Extremos rotulados con su signo real
    for k in sorted({int(np.argmax(vs)), int(np.argmin(vs))}):
        if abs(vs[k]) < 1e-9 * max(float(np.abs(vs).max()), 1e-12):
            continue
        px, py = P(float(xs[k]), float(plot[k]))
        up = plot[k] >= 0
        d.add(Circle(px, py, 1.6, fillColor=color, strokeColor=None))
        _label(d, px, py + (3.5 if up else -9.5), _short(float(vs[k])), 7.0, color, bold=True)
    return d


def _fig_frame_diagram(model: Model, results: Results, q: str, w: float) -> Drawing:
    """Diagrama dibujado perpendicular a cada barra sobre la geometría (pórticos)."""
    _, _, f = _DIAG_SPEC[q]
    color = C_DIAG[q]
    off_max = 22.0
    pad = (off_max + 12.0,) * 4
    h = _auto_height(model, w, pad, 120.0, 200.0)
    d = Drawing(w, h)
    view = _fit(model.nodes, w, h, pad)
    vmax = max(float(np.abs(_values(mr, q)).max()) for mr in results.members) or 1.0
    scale = off_max / vmax
    _draw_members(d, model, view, C_MEMBER, 1.6)
    labels: list[tuple[Pt, float]] = []
    for mr in results.members:
        c, s = model.member_cos_sin(mr.member)
        ny = (-s, c)
        v = _values(mr, q)
        sgn = -1.0 if q == "M" else 1.0  # M del lado traccionado
        base = [view(float(p[0]), float(p[1])) for p in mr.points]
        off = [(bx + sgn * float(vi) * scale * ny[0], by + sgn * float(vi) * scale * ny[1])
               for (bx, by), vi in zip(base, v, strict=True)]
        ring = [base[0], *off, base[-1]]
        d.add(Polygon([cc for p in ring for cc in p], fillColor=_tint(color, 0.18), strokeColor=None))
        _poly(d, off, color, 1.3)
        for k in sorted({0, len(v) - 1, int(np.argmax(np.abs(v)))}):
            if abs(v[k]) > 0.03 * vmax:
                labels.append((off[k], float(v[k]) * f))
    _draw_supports(d, model, view)
    _draw_local_axes(d, model, view, t=0.42, L=11.0, with_y=False)
    placed: list[Pt] = []
    for (x, y), val in labels:  # evita rótulos superpuestos en nudos compartidos
        if any(math.hypot(x - a, y - b) < 14 for a, b in placed):
            continue
        placed.append((x, y))
        _label(d, x, y - 2.5, _short(val), 6.8, color, bold=True)
    return d


def _diagram_figures(model: Model, results: Results, figs: _Figures) -> list[Flowable]:
    out: list[Flowable] = []
    beam = model.is_horizontal_beam()
    shown: list[str] = []
    for q in ("M", "V", "N", "sigma"):
        vmax = max(float(np.abs(_values(mr, q)).max()) for mr in results.members) * _DIAG_SPEC[q][2]
        if vmax < 5e-4:  # nulo a la precisión impresa (0.000 en las tablas)
            label = _DIAG_SPEC[q][0]
            out.append(_p(f"{label}: nulo en toda la estructura (no se grafica).", "note"))
            continue
        shown.append(q)

    def cap(q: str) -> str:
        title, unit, _ = _DIAG_SPEC[q]
        title = (title[0].lower() + title[1:]).replace("σ", _g("s"))
        extra = " Dibujado del lado traccionado (+ hacia abajo en vigas)." if q == "M" else ""
        if not beam and q in ("N", "V"):
            extra += " Signo según ejes locales x' (de i a j) de cada barra."
        return f"Diagrama de {title} [{unit}].{extra}"

    if beam:
        for q in shown:
            out.append(figs.add(_fig_beam_diagram(model, results, q, CONTENT_W), cap(q)))
        return out
    # Pórtico: grilla de 2 columnas
    cw = CONTENT_W / 2
    cells = [[_fig_frame_diagram(model, results, q, cw - 4), figs.caption(cap(q))] for q in shown]
    rows = [cells[i:i + 2] + ([[""]] if len(cells[i:i + 2]) == 1 else []) for i in range(0, len(cells), 2)]
    for row in rows:
        t = Table([row], colWidths=[cw, cw])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Helpers de maquetación
# --------------------------------------------------------------------------- #


_GREEK = {"s": "σ", "d": "δ", "n": "ν", "r": "ρ"}


def _g(letter: str) -> str:
    """Letra griega para Paragraph: 's' -> σ, 'd' -> δ (vía fuente Symbol si no hay DejaVu)."""
    return _GREEK[letter] if _UNICODE else f'<font face="Symbol">{_GREEK[letter]}</font>'


def _gt(letter: str) -> str:
    """Letra griega para texto plano de figuras (sin markup)."""
    return _GREEK[letter] if _UNICODE else {"s": "sigma", "d": "delta"}.get(letter, letter)


_TO = "→" if _UNICODE else "->"


def _sup(n: int) -> str:
    """Exponente con markup <super> (uniforme para ², ³ y ⁴)."""
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
