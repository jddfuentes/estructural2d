"""Materiales y catálogo de perfiles estándar.

Unidades SI-mm: longitudes en mm, áreas en mm², inercias en mm⁴,
módulos resistentes en mm³, tensiones y módulo elástico en MPa (N/mm²),
densidad en kg/m³.

Los valores tabulados de perfiles laminados provienen de tablas de catálogo
usuales: IPE / IPN (Euronorm 19-57 / DIN 1025), UPN (DIN 1026-1) y W
(ASTM A6/A6M, AISC Shapes Database v15.0). Los tubos y los perfiles C de chapa
doblada se calculan a partir de su geometría nominal (tubos con esquinas vivas,
conservador; perfiles C con radio interior de plegado). Verificar siempre contra
el catálogo del proveedor antes de un cálculo final.

Todas las secciones se usan en flexión alrededor del eje fuerte (x-x). UPN y C
son simétricos respecto de ese eje (c_top = c_bot = h/2), pero su centro de
corte no coincide con el baricentro: el modelo no considera la torsión que
aparece si la carga no pasa por el centro de corte (limitación de
predimensionamiento, ver AGENTS.md §10).
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


def rolled_channel(name: str, h: float, b: float, A_cm2: float, I_cm4: float) -> Section:
    """Perfil laminado U (UPN) en eje fuerte a partir de valores de tabla (cm², cm⁴).

    Simétrico respecto del eje x-x, por lo que c_top = c_bot = h/2.
    """
    return Section(
        name=name, family="UPN", A=A_cm2 * 100.0, I=I_cm4 * 1e4,
        c_top=h / 2.0, c_bot=h / 2.0, h=h, b=b,
    )


def wide_flange(name: str, d: float, bf: float, A: float, I: float) -> Section:  # noqa: E741
    """Perfil W (ala ancha, ASTM A6) a partir de valores de tabla ya en mm, mm², mm⁴.

    `d` es la altura real del perfil (no la nominal de la designación).
    """
    return Section(
        name=name, family="Perfil W", A=A, I=I,
        c_top=d / 2.0, c_bot=d / 2.0, h=d, b=bf,
    )


def cold_formed_channel(name: str, H: float, B: float, t: float, r_i: float | None = None) -> Section:
    """Perfil C de chapa doblada (canal sin labios), propiedades geométricas exactas.

    H = altura exterior del alma, B = ancho exterior del ala, t = espesor,
    r_i = radio interior de plegado (por defecto r_i = t, práctica usual de plegado
    en acero al carbono). La sección se descompone en tramos rectos (alma y alas)
    y cuatro esquinas en cuarto de corona circular, integradas en forma cerrada:
      corona (r_i, r_o = r_i + t): A_c = π/4·(r_o² − r_i²),
      I propia respecto del centro de curvatura = π/16·(r_o⁴ − r_i⁴),
      momento estático respecto de ese centro Q = (r_o³ − r_i³)/3,
      traslado al eje x-x: I = I_c + 2·d·Q + d²·A_c, con d = H/2 − r_o.
    """
    r_in = t if r_i is None else r_i
    r_o = r_in + t
    if not (t > 0.0 and r_in >= 0.0 and H > 2.0 * r_o and B > r_o):
        raise ValueError(f"Geometría inválida para el perfil C {name!r}: revisar H, B, t y r_i.")
    # Alma (tramo recto, centrado en el eje x-x)
    hw = H - 2.0 * r_o
    A_web = t * hw
    I_web = t * hw**3 / 12.0
    # Alas (tramo recto), una arriba y otra abajo
    bf = B - r_o
    y_f = H / 2.0 - t / 2.0
    A_fl = t * bf
    I_fl = bf * t**3 / 12.0 + A_fl * y_f**2
    # Esquinas (cuarto de corona), centro de curvatura a d = H/2 − r_o del eje x-x
    A_c = math.pi / 4.0 * (r_o**2 - r_in**2)
    I_c0 = math.pi / 16.0 * (r_o**4 - r_in**4)
    Q_c0 = (r_o**3 - r_in**3) / 3.0
    d = H / 2.0 - r_o
    I_c = I_c0 + 2.0 * d * Q_c0 + d**2 * A_c
    return Section(
        name=name, family="Perfil C (Conformado)",
        A=A_web + 2.0 * (A_fl + A_c), I=I_web + 2.0 * (I_fl + I_c),
        c_top=H / 2.0, c_bot=H / 2.0, h=H, b=B,
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

# UPN — DIN 1026-1 (EN 10279): (h [mm], b [mm], A [cm²], Ix [cm⁴])
_UPN: dict[int, tuple[float, float, float, float]] = {
    80: (80, 45, 11.0, 106.0),
    100: (100, 50, 13.5, 206.0),
    120: (120, 55, 17.0, 364.0),
    140: (140, 60, 20.4, 605.0),
    160: (160, 65, 24.0, 925.0),
    180: (180, 70, 28.0, 1350.0),
    200: (200, 75, 32.2, 1910.0),
    220: (220, 80, 37.4, 2690.0),
    240: (240, 85, 42.3, 3600.0),
    260: (260, 90, 48.3, 4820.0),
    280: (280, 95, 53.3, 6280.0),
    300: (300, 100, 58.8, 8030.0),
}

# Perfiles W — ASTM A6/A6M, AISC Shapes Database v15.0. Valores de la tabla en
# unidades US convertidos a SI (1 in = 25,4 mm) y redondeados; equivalente US en
# el comentario. (d [mm], bf [mm], A [mm²], Ix [mm⁴]); d = altura real.
_W: dict[str, tuple[float, float, float, float]] = {
    "W 150x13": (148.1, 100.1, 1626.0, 6.202e6),  # W6x8.5
    "W 150x18": (153.2, 101.6, 2290.0, 9.199e6),  # W6x12
    "W 200x15": (200.4, 100.1, 1910.0, 12.82e6),  # W8x10
    "W 200x22.5": (206.0, 102.0, 2865.0, 19.98e6),  # W8x15
    "W 250x28.4": (260.1, 102.1, 3626.0, 40.08e6),  # W10x19
    "W 310x38.7": (310.4, 164.8, 4935.0, 84.91e6),  # W12x26
}

# Perfiles C de chapa doblada (canal sin labios), geometría nominal de la
# designación C H x B x t: (H [mm], B [mm], t [mm]); r_i = t (ver cold_formed_channel).
_C_COLD: tuple[tuple[float, float, float], ...] = (
    (80, 40, 2.0),
    (100, 50, 2.0),
    (120, 50, 2.0),
    (140, 60, 2.5),
    (160, 60, 2.5),
    (200, 75, 3.0),
)

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
    for n, (h, b, A, I) in _UPN.items():
        add(rolled_channel(f"UPN {n}", h, b, A, I))
    for name, (d, bf, A, I) in _W.items():
        add(wide_flange(name, d, bf, A, I))
    for H, B, t in _C_COLD:
        add(cold_formed_channel(f"C {H:g}x{B:g}x{t:g}", H, B, t))
    for nps, D, t in _PIPE_SCH40:
        add(chs(f"Caño {nps} Sch40 (Ø{D:g}x{t:g})", D, t))
    for B, H, t in _RHS:
        add(rhs(f"Tubo {H:g}x{B:g}x{t:g}", B, H, t))
    return cat


SECTIONS: dict[str, dict[str, Section]] = _build_catalog()
"""Catálogo: familia -> nombre -> Section."""

DEFAULT_SECTION = ("IPE", "IPE 200")


# ---- Notas y advertencias por familia ----------------------------------------- #
# Fuente única del criterio de ingeniería asociado a cada familia: la UI y la memoria
# PDF muestran estos textos tal cual y no redactan advertencias propias (AGENTS.md §7.2).

FAMILY_NOTES: dict[str, str] = {
    "IPE": "Doble T laminado IPE (Euronorm 19-57). Propiedades nominales de tabla.",
    "IPN": "Doble T laminado IPN, alas inclinadas (DIN 1025-1). Propiedades nominales de tabla.",
    "UPN": "Canal laminado UPN (DIN 1026-1 / EN 10279), flexión en eje fuerte x-x. "
           "Propiedades nominales de tabla.",
    "Perfil W": "Ala ancha ASTM A6/A6M (AISC Shapes Database v15.0, convertido a SI). "
                "La designación W d×m es altura nominal [mm] × masa [kg/m]; h es la altura real.",
    "Perfil C (Conformado)": "Chapa doblada en frío, canal sin labios C H×B×t con radio interior "
                             "de plegado igual al espesor. Propiedades brutas calculadas de la "
                             "geometría nominal, no de catálogo de fabricante.",
    "Tubo circular": "Caño ASME B36.10 Sch 40 / API 5L. Propiedades calculadas de D y t nominales.",
    "Tubo cuadrado": "Tubo estructural. Propiedades calculadas de B, H y t con esquinas vivas.",
    "Tubo rectangular": "Tubo estructural, flexión alrededor del eje de mayor altura H. Propiedades "
                        "calculadas de B, H y t con esquinas vivas.",
    "Macizo": "Barra maciza. Propiedades calculadas de la geometría.",
    "Personalizado": "Sección simétrica definida por el usuario: A, I y c son responsabilidad de "
                     "quien los ingresa.",
}

CHANNEL_FAMILIES: frozenset[str] = frozenset({"UPN", "Perfil C (Conformado)"})
"""Familias de sección en canal (U/C): monosimétricas, centro de corte fuera del alma."""

THIN_WALLED_FAMILIES: frozenset[str] = frozenset({"Perfil C (Conformado)"})
"""Familias de chapa delgada conformada en frío, donde la abolladura local puede gobernar."""

CHANNEL_TORSION_WARNING = (
    "Sección en canal (U/C): el centro de corte queda del lado exterior del alma. Si la carga "
    "no pasa por él, la pieza se tuerce y aparecen tensiones que este modelo no calcula. "
    "Usar perfiles apareados (cajón o espalda con espalda) o restringir el giro."
)

THIN_WALL_WARNING = (
    "Chapa conformada en frío: se usan propiedades brutas. La abolladura local de alas y alma "
    "(ancho efectivo, AISI S100 / CIRSOC 303) puede reducir la capacidad real; el FS informado "
    "no la considera."
)


def family_warnings(family: str) -> tuple[str, ...]:
    """Advertencias de alcance que la UI y la memoria deben mostrar para una familia."""
    out: list[str] = []
    if family in CHANNEL_FAMILIES:
        out.append(CHANNEL_TORSION_WARNING)
    if family in THIN_WALLED_FAMILIES:
        out.append(THIN_WALL_WARNING)
    return tuple(out)


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
