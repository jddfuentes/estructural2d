"""Materiales y catálogo de perfiles estándar.

Unidades SI-mm: longitudes en mm, áreas en mm², inercias en mm⁴,
módulos resistentes en mm³, tensiones y módulo elástico en MPa (N/mm²),
densidad en kg/m³.

Los valores tabulados de perfiles laminados (IPE / IPN) provienen de tablas
de catálogo usuales (Euronorm 19-57 / DIN 1025). Los tubos se calculan a
partir de su geometría nominal (esquinas vivas para RHS/SHS, conservador).
Verificar siempre contra el catálogo del proveedor antes de un cálculo final.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Materiales
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Material:
    """Material isótropo lineal-elástico."""

    name: str
    E: float  # Módulo de elasticidad [MPa]
    Sy: float  # Tensión de fluencia [MPa]
    Su: float  # Tensión de rotura [MPa]
    nu: float = 0.3  # Coeficiente de Poisson [-]
    rho: float = 7850.0  # Densidad [kg/m³]

    @property
    def G(self) -> float:
        """Módulo de corte [MPa]."""
        return self.E / (2.0 * (1.0 + self.nu))


MATERIALS: dict[str, Material] = {
    m.name: m
    for m in (
        Material("ASTM A36", E=200_000.0, Sy=250.0, Su=400.0),
        Material("F-24 (IRAM-IAS U500-503)", E=200_000.0, Sy=235.0, Su=370.0),
        Material("F-36 (IRAM-IAS U500-503)", E=200_000.0, Sy=355.0, Su=490.0),
        Material("ASTM A572 Gr.50", E=200_000.0, Sy=345.0, Su=450.0),
        Material("API 5L X52", E=207_000.0, Sy=359.0, Su=455.0),
        Material("SAE 1045 (laminado)", E=205_000.0, Sy=310.0, Su=565.0),
        Material("Al 6061-T6", E=68_900.0, Sy=276.0, Su=310.0, nu=0.33, rho=2700.0),
    )
}

DEFAULT_MATERIAL = "ASTM A36"


# --------------------------------------------------------------------------- #
# Secciones
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Section:
    """Sección transversal, flexión alrededor del eje fuerte.

    `c_top` y `c_bot` son las distancias del eje neutro a las fibras extremas
    (iguales en secciones simétricas). Se mantienen separadas para admitir
    secciones asimétricas a futuro (T, canales flexionando en eje débil, etc.).
    """

    name: str
    family: str
    A: float  # Área [mm²]
    I: float  # Momento de inercia eje fuerte [mm⁴]  # noqa: E741
    c_top: float  # Distancia a fibra superior [mm]
    c_bot: float  # Distancia a fibra inferior [mm]
    h: float  # Altura total [mm]
    b: float = 0.0  # Ancho [mm] (informativo)

    @property
    def c_max(self) -> float:
        return max(self.c_top, self.c_bot)

    @property
    def W(self) -> float:
        """Módulo resistente elástico mínimo [mm³] = I / c_max."""
        return self.I / self.c_max

    @property
    def mass_per_m(self) -> float:
        """Masa lineal para acero (ρ = 7850 kg/m³) [kg/m]."""
        return self.A * 1e-6 * 7850.0


# ---- Constructores ---------------------------------------------------------- #


def rolled_i(name: str, family: str, h: float, b: float, A_cm2: float, I_cm4: float) -> Section:
    """Perfil laminado doble T simétrico a partir de valores de tabla (cm², cm⁴)."""
    return Section(
        name=name, family=family, A=A_cm2 * 100.0, I=I_cm4 * 1e4,
        c_top=h / 2.0, c_bot=h / 2.0, h=h, b=b,
    )


def chs(name: str, D: float, t: float) -> Section:
    """Tubo circular (CHS / caño)."""
    d = D - 2.0 * t
    return Section(
        name=name, family="Tubo circular",
        A=math.pi / 4.0 * (D**2 - d**2), I=math.pi / 64.0 * (D**4 - d**4),
        c_top=D / 2.0, c_bot=D / 2.0, h=D, b=D,
    )


def rhs(name: str, B: float, H: float, t: float) -> Section:
    """Tubo rectangular / cuadrado (esquinas vivas). H = altura en el plano de flexión."""
    family = "Tubo cuadrado" if math.isclose(B, H) else "Tubo rectangular"
    return Section(
        name=name, family=family,
        A=B * H - (B - 2.0 * t) * (H - 2.0 * t),
        I=(B * H**3 - (B - 2.0 * t) * (H - 2.0 * t) ** 3) / 12.0,
        c_top=H / 2.0, c_bot=H / 2.0, h=H, b=B,
    )


def solid_rect(name: str, b: float, h: float) -> Section:
    """Barra rectangular maciza / planchuela."""
    return Section(
        name=name, family="Macizo", A=b * h, I=b * h**3 / 12.0,
        c_top=h / 2.0, c_bot=h / 2.0, h=h, b=b,
    )


def solid_round(name: str, d: float) -> Section:
    """Barra redonda maciza."""
    return Section(
        name=name, family="Macizo", A=math.pi * d**2 / 4.0, I=math.pi * d**4 / 64.0,
        c_top=d / 2.0, c_bot=d / 2.0, h=d, b=d,
    )


def custom(name: str, A: float, I: float, c: float) -> Section:  # noqa: E741
    """Sección definida por el usuario (simétrica)."""
    return Section(name=name, family="Personalizado", A=A, I=I, c_top=c, c_bot=c, h=2.0 * c)


# ---- Catálogo ---------------------------------------------------------------- #

# (h [mm], b [mm], A [cm²], Iy [cm⁴])
_IPE: dict[int, tuple[float, float, float, float]] = {
    80: (80, 46, 7.64, 80.1),
    100: (100, 55, 10.3, 171.0),
    120: (120, 64, 13.2, 318.0),
    140: (140, 73, 16.4, 541.0),
    160: (160, 82, 20.1, 869.0),
    180: (180, 91, 23.9, 1317.0),
    200: (200, 100, 28.5, 1943.0),
    220: (220, 110, 33.4, 2772.0),
    240: (240, 120, 39.1, 3892.0),
    270: (270, 135, 45.9, 5790.0),
    300: (300, 150, 53.8, 8356.0),
    330: (330, 160, 62.6, 11770.0),
    360: (360, 170, 72.7, 16270.0),
    400: (400, 180, 84.5, 23130.0),
    450: (450, 190, 98.8, 33740.0),
    500: (500, 200, 116.0, 48200.0),
    600: (600, 220, 156.0, 92080.0),
}

_IPN: dict[int, tuple[float, float, float, float]] = {
    80: (80, 42, 7.57, 77.8),
    100: (100, 50, 10.6, 171.0),
    120: (120, 58, 14.2, 328.0),
    140: (140, 66, 18.2, 573.0),
    160: (160, 74, 22.8, 935.0),
    180: (180, 82, 27.9, 1450.0),
    200: (200, 90, 33.4, 2140.0),
    220: (220, 98, 39.5, 3060.0),
    240: (240, 106, 46.1, 4250.0),
    260: (260, 113, 53.3, 5740.0),
    280: (280, 119, 61.0, 7590.0),
    300: (300, 125, 69.0, 9800.0),
}

# Caños API 5L / ASME B36.10 Sch 40: (NPS, D [mm], t [mm])
_PIPE_SCH40: tuple[tuple[str, float, float], ...] = (
    ('1"', 33.4, 3.38),
    ('2"', 60.3, 3.91),
    ('3"', 88.9, 5.49),
    ('4"', 114.3, 6.02),
    ('6"', 168.3, 7.11),
    ('8"', 219.1, 8.18),
    ('10"', 273.0, 9.27),
)

# Tubos estructurales: (B, H, t) en mm
_RHS: tuple[tuple[float, float, float], ...] = (
    (40, 40, 2.0), (50, 50, 3.2), (60, 60, 3.2), (80, 80, 4.0), (100, 100, 5.0),
    (120, 120, 6.0), (150, 150, 6.0),
    (60, 40, 2.0), (80, 40, 3.2), (100, 50, 3.2), (120, 60, 4.0), (150, 100, 5.0),
    (200, 100, 6.0),
)


def _build_catalog() -> dict[str, dict[str, Section]]:
    cat: dict[str, dict[str, Section]] = {}

    def add(s: Section) -> None:
        cat.setdefault(s.family, {})[s.name] = s

    for n, (h, b, A, I) in _IPE.items():
        add(rolled_i(f"IPE {n}", "IPE", h, b, A, I))
    for n, (h, b, A, I) in _IPN.items():
        add(rolled_i(f"IPN {n}", "IPN", h, b, A, I))
    for nps, D, t in _PIPE_SCH40:
        add(chs(f"Caño {nps} Sch40 (Ø{D:g}x{t:g})", D, t))
    for B, H, t in _RHS:
        add(rhs(f"Tubo {H:g}x{B:g}x{t:g}", B, H, t))
    return cat


SECTIONS: dict[str, dict[str, Section]] = _build_catalog()
"""Catálogo: familia -> nombre -> Section."""

DEFAULT_SECTION = ("IPE", "IPE 200")


def get_section(name: str) -> Section:
    """Busca un perfil por nombre en todas las familias."""
    for family in SECTIONS.values():
        if name in family:
            return family[name]
    raise KeyError(f"Perfil no encontrado en el catálogo: {name!r}")


def find_lightest(min_I: float = 0.0, min_W: float = 0.0, family: str | None = None) -> Section | None:
    """Perfil de menor área que cumple I >= min_I y W >= min_W (predimensionamiento)."""
    pool = (
        SECTIONS[family].values() if family else (s for fam in SECTIONS.values() for s in fam.values())
    )
    ok = [s for s in pool if s.I >= min_I and s.W >= min_W]
    return min(ok, key=lambda s: s.A) if ok else None
