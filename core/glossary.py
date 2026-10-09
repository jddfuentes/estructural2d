"""Glosario y simbología: fuente única para la memoria PDF y la wiki de la app.

Datos puros (sin dependencias). Los textos están en Unicode con una mini-notación:
    base_{sub}  -> subíndice       (σ_{máx}, FS_{adm}, c_{máx})
    base^{sup}  -> superíndice     (cm^{2}, cm^{4})

Consumidores:
    core.reports      -> sección "6. Glosario y simbología" del PDF (convierte a markup ReportLab).
    to_markdown()     -> Markdown con símbolos en LaTeX inline ($...$), para `st.markdown` en la app.

Regla (AGENTS.md): todo símbolo, ícono o término nuevo que aparezca en la memoria o en la app se
agrega acá; ni el PDF ni la UI definen glosario propio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "GRAPHIC_SYMBOLS",
    "NOTATION",
    "TERMS",
    "Entry",
    "GraphicSymbol",
    "Group",
    "Term",
    "to_markdown",
    "to_tex",
]


@dataclass(frozen=True, slots=True)
class Entry:
    symbol: str
    description: str
    unit: str = "—"


@dataclass(frozen=True, slots=True)
class Group:
    title: str
    entries: tuple[Entry, ...]


@dataclass(frozen=True, slots=True)
class GraphicSymbol:
    kind: str  # clave del ícono que dibuja core.reports (_icon)
    name: str
    meaning: str


@dataclass(frozen=True, slots=True)
class Term:
    term: str
    definition: str


# --------------------------------------------------------------------------- #
# Notación
# --------------------------------------------------------------------------- #

NOTATION: tuple[Group, ...] = (
    Group(
        "Ejes y geometría",
        (
            Entry("X, Y", "Ejes globales: X hacia la derecha, Y hacia arriba."),
            Entry("x', y'", "Ejes locales de barra: x' del nodo i al nodo j; y' = x' girado +90°."),
            Entry("i, j", "Nodo inicial y nodo final de una barra."),
            Entry("L", "Longitud de barra; en L/δ, longitud de referencia (luz).", "m"),
            Entry("h", "Altura total de la sección transversal.", "mm"),
            Entry(
                "c, c_{máx}",
                "Distancia del eje neutro a la fibra extrema; c_{máx} es la mayor de ambas.",
                "mm",
            ),
        ),
    ),
    Group(
        "Cargas y reacciones",
        (
            Entry(
                "F_{x}, F_{y}",
                "Fuerza concentrada en un nodo, en ejes globales (F_{y} < 0: hacia abajo).",
                "kN",
            ),
            Entry("q", "Carga distribuida por unidad de longitud de barra (q < 0: hacia abajo).", "kN/m"),
            Entry(
                "M_{z}", "Momento concentrado aplicado o reacción de momento; positivo antihorario.", "kN·m"
            ),
            Entry(
                "R_{x}, R_{y}", "Reacciones de vínculo en ejes globales, actuando sobre la estructura.", "kN"
            ),
            Entry("Σ", "Sumatoria. Equilibrio global: ΣR + ΣF = 0."),
        ),
    ),
    Group(
        "Solicitaciones",
        (
            Entry("N", "Esfuerzo normal; positivo de tracción.", "kN"),
            Entry("V", "Esfuerzo de corte; V = dM/dx (par horario positivo sobre el elemento).", "kN"),
            Entry("M", "Momento flector; positivo si tracciona la fibra inferior (y' negativa).", "kN·m"),
        ),
    ),
    Group(
        "Desplazamientos",
        (
            Entry("u_{x}, u_{y}", "Desplazamientos nodales según X e Y.", "mm"),
            Entry("θ_{z}", "Giro nodal alrededor de Z; positivo antihorario.", "rad"),
            Entry(
                "δ, δ_{máx}",
                "Desplazamiento total (módulo de u_{x} y u_{y}) y su máximo en la estructura.",
                "mm",
            ),
            Entry("L/δ", "Flecha relativa: longitud de referencia dividida por δ_{máx}."),
        ),
    ),
    Group(
        "Sección y material",
        (
            Entry("A", "Área de la sección transversal.", "cm^{2}"),
            Entry("I", "Momento de inercia respecto del eje de flexión.", "cm^{4}"),
            Entry("W", "Módulo resistente elástico, W = I / c_{máx}.", "cm^{3}"),
            Entry("E", "Módulo de elasticidad longitudinal.", "MPa"),
            Entry("S_{y}, S_{u}", "Tensión de fluencia y tensión de rotura del material.", "MPa"),
            Entry("ν", "Coeficiente de Poisson."),
            Entry("ρ", "Densidad del material (define el peso propio).", "kg/m^{3}"),
        ),
    ),
    Group(
        "Verificación",
        (
            Entry(
                "σ, |σ|_{máx}",
                "Tensión normal en fibra extrema, σ = N/A ± M·c/I, y su máximo absoluto en la estructura.",
                "MPa",
            ),
            Entry("FS", "Factor de seguridad a fluencia, FS = S_{y} / σ_{máx}."),
            Entry("FS_{adm}", "Factor de seguridad mínimo exigido (admisible)."),
            Entry("Aprov.", "Aprovechamiento = σ_{máx} · FS_{adm} / S_{y}; más de 100 % no verifica.", "%"),
            Entry("GDL", "Grado de libertad nodal (u_{x}, u_{y}, θ_{z})."),
        ),
    ),
)

# --------------------------------------------------------------------------- #
# Símbolos gráficos (los íconos los dibuja core.reports según `kind`)
# --------------------------------------------------------------------------- #

GRAPHIC_SYMBOLS: tuple[GraphicSymbol, ...] = (
    GraphicSymbol("fixed", "Empotrado", "Apoyo que restringe u_{x}, u_{y} y θ_{z} (traslaciones y giro)."),
    GraphicSymbol("pinned", "Articulado", "Apoyo que restringe u_{x} y u_{y}; permite el giro."),
    GraphicSymbol("roller", "Móvil", "Apoyo que restringe sólo u_{y}; permite desplazamiento en X y giro."),
    GraphicSymbol(
        "point", "Carga concentrada", "Fuerza aplicada en un nodo; la flecha indica su sentido real."
    ),
    GraphicSymbol(
        "dist",
        "Carga distribuida",
        "Carga por unidad de longitud; la altura es proporcional a q (variación lineal entre extremos).",
    ),
    GraphicSymbol(
        "moment", "Momento aplicado", "Momento concentrado en un nodo; la flecha indica el sentido de giro."
    ),
    GraphicSymbol(
        "global", "Ejes globales", "Terna X-Y de referencia y sentido positivo de M_{z} (antihorario)."
    ),
    GraphicSymbol(
        "local", "Ejes locales", "x' sobre la barra de i a j, y' a +90°. Referencia del signo de N y V."
    ),
    GraphicSymbol(
        "deformed", "Deformada", "En azul, la deformada amplificada; en gris, la geometría sin deformar."
    ),
    GraphicSymbol(
        "diagram", "Diagrama", "Área coloreada proporcional a la solicitación (M del lado traccionado)."
    ),
    GraphicSymbol("dim", "Cota", "Distancia entre puntos de la estructura, en m."),
)

# --------------------------------------------------------------------------- #
# Términos
# --------------------------------------------------------------------------- #

TERMS: tuple[Term, ...] = (
    Term(
        "Predimensionamiento",
        "Selección preliminar de perfiles con un modelo simplificado, previa al "
        "cálculo reglamentario definitivo.",
    ),
    Term(
        "Método directo de rigidez",
        "Ensambla K·u = F con las matrices de rigidez de cada barra, resuelve los "
        "desplazamientos nodales y de ellos obtiene reacciones y solicitaciones.",
    ),
    Term(
        "Euler-Bernoulli",
        "Teoría de vigas que desprecia la deformación por corte: las secciones planas "
        "permanecen planas y normales al eje deformado.",
    ),
    Term(
        "Lado traccionado",
        "Convención de dibujo del diagrama de M: la ordenada se traza del lado de la "
        "fibra traccionada de la barra.",
    ),
    Term(
        "Deformada amplificada",
        "Geometría deformada con los desplazamientos multiplicados por un factor de "
        "escala para hacerlos visibles; los valores informados no se escalan.",
    ),
    Term(
        "Peso propio",
        "Carga vertical distribuida igual a ρ·g·A del perfil. Se incluye en el cálculo y no se "
        "dibuja en el esquema.",
    ),
    Term(
        "Fluencia",
        "Estado en que la tensión alcanza S_{y}. El veredicto FLUENCIA indica FS < 1; ALERTA, "
        "1 ≤ FS < FS_{adm}.",
    ),
)

# --------------------------------------------------------------------------- #
# Render a Markdown (app) — símbolos en LaTeX inline, compatible con st.markdown
# --------------------------------------------------------------------------- #

_TEX_CHARS = {
    "σ": r"\sigma ",
    "δ": r"\delta ",
    "ν": r"\nu ",
    "ρ": r"\rho ",
    "θ": r"\theta ",
    "Σ": r"\Sigma ",
    "|": r"\vert ",
    "·": r"\cdot ",
    "%": r"\%",
}
_SCRIPT = re.compile(r"([_^])\{([^}]*)\}")
_TOKEN = re.compile(r"([^\s(,;]+?)([_^])\{([^}]*)\}")  # palabra con sub/superíndice dentro de un texto
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def to_tex(symbol: str) -> str:
    """Símbolo en mini-notación -> LaTeX (sin delimitadores). 'σ_{máx}' -> '\\sigma _{\\text{máx}}'."""

    def script(m: re.Match[str]) -> str:
        body = m.group(2)
        inner = body if body.isdigit() or len(body) == 1 else rf"\text{{{body}}}"
        return f"{m.group(1)}{{{inner}}}"

    out = _SCRIPT.sub(script, symbol)
    out = "".join(_TEX_CHARS.get(ch, ch) for ch in out)
    # palabras de más de una letra fuera de sub/superíndices, en texto recto (FS, GDL, Aprov.)
    return re.sub(r"(?<![\\{a-zA-Z])([A-Z]{2,}\.?|Aprov\.)", r"\\text{\1}", out)


def _md_text(text: str) -> str:
    """Texto corrido: tokens con sub/superíndice -> $LaTeX$; resto queda como texto."""
    return _TOKEN.sub(lambda m: f"${to_tex(m.group(0))}$", text)


def _md_unit(unit: str) -> str:
    return _SCRIPT.sub(
        lambda m: m.group(2).translate(_SUPERSCRIPT_DIGITS) if m.group(1) == "^" else m.group(2), unit
    )


def to_markdown(include_graphics: bool = True, heading_level: int = 2) -> str:
    """Glosario completo en Markdown (tablas GFM + LaTeX inline) para `st.markdown`."""
    h = "#" * heading_level
    lines = [f"{h} Glosario y simbología", "", f"{h}# Notación", ""]
    for group in NOTATION:
        lines += [f"**{group.title}**", "", "| Símbolo | Descripción | Unidad |", "|:--|:--|--:|"]
        lines += [
            f"| ${to_tex(e.symbol)}$ | {_md_text(e.description)} | {_md_unit(e.unit)} |"
            for e in group.entries
        ]
        lines.append("")
    if include_graphics:
        lines += [
            f"{h}# Símbolos gráficos",
            "",
            "Los íconos se dibujan en la memoria PDF (sección 6.2).",
            "",
            "| Símbolo | Significado |",
            "|:--|:--|",
        ]
        lines += [f"| **{g.name}** | {_md_text(g.meaning)} |" for g in GRAPHIC_SYMBOLS]
        lines.append("")
    lines += [f"{h}# Términos", "", "| Término | Definición |", "|:--|:--|"]
    lines += [f"| **{t.term}** | {_md_text(t.definition)} |" for t in TERMS]
    return "\n".join(lines) + "\n"
