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

Por defecto las secciones flexionan alrededor del eje fuerte (x-x). UPN y C
son simétricos respecto de ese eje (c_top = c_bot = h/2), pero su centro de
corte no coincide con el baricentro: el modelo no considera la torsión que
aparece si la carga no pasa por el centro de corte (limitación de
predimensionamiento, ver AGENTS.md §10).

Corte y eje débil: cada constructor calcula además Q_max y t_web (tensión de corte
τ = V·Q/(I·t) en el eje neutro) y las propiedades de eje débil (Iy, Wy, distancias a
fibras extremas c_y1/c_y2, Q_y, t_y). `Section.weak_axis()` devuelve la misma sección
orientada para flexionar en su eje débil dentro del plano del pórtico; en canales
(UPN, C) es asimétrica: c_top = e (dorso del alma) ≠ c_bot = b − e (puntas de ala).
- IPE y W (alas paralelas con radio de acuerdo): Q, Iy por geometría (h, b, tw, tf) más
  los cuatro acuerdos; el radio r se deduce del área de tabla: 4·(1 − π/4)·r² = A − A_rect.
  Reproduce Wpl/2 e Iz de tabla con error < 1 %.
- IPN y UPN (alas inclinadas, tf = espesor medio de tabla): Q por geometría con tf medio
  (≈ S_x de tabla); Iy y, en UPN, e = distancia del baricentro al dorso del alma, de tabla.

Estabilidad (pandeo flexional y lateral-torsional, AISC 360-16 / CIRSOC 301-2005): radios de giro
r_x = √(I/A), r_y = √(Iy/A); módulo plástico Z = 2·Q_max en secciones simétricas respecto del eje de
flexión (exacto: el eje neutro plástico coincide con el elástico). Doble T y canales llevan además J
(torsión de Saint-Venant), Cw (alabeo) y h0 (distancia entre baricentros de alas):
- IPE y W: J = 2/3·b·tf³ + 1/3·(h − 2tf)·tw³ + 2·α·D⁴ − 0,420·tf⁴ (El Darwish y Johnston, 1965; usada en
  las tablas ArcelorMittal y en AISC Design Guide 9 §3.2), con el radio r de acuerdo deducido del área;
  reproduce It de tabla con error < 1 %. Cw = Iy·h0²/4, h0 = h − tf.
- IPN: J por rectángulos sin acuerdos (≈ 15 % menor que tabla, conservador); Cw = Iy·h0²/4 con Iy de
  tabla (reproduce Iw de tabla con error < 1 %).
- UPN y C conformado (pared delgada, AISC Design Guide 9 §3.2): b' = b − tw/2, h0 = h − tf,
  J = (2·b'·tf³ + h0·tw³)/3, Cw = tf·b'³·h0²/12 · (3·b'·tf + 2·h0·tw)/(6·b'·tf + h0·tw). En UPN, con tf
  medio, Cw resulta ≈ 15 % mayor que el de tabla (alas en cuña): aproximación de predimensionamiento.
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
    # Coeficiente de dilatación térmica lineal [1/°C]. Acero estructural: 12·10⁻⁶ 1/°C
    # (EN 1993-1-1 §3.2.6; mismo valor usual en CIRSOC 301). Agregado al final (AGENTS.md §5).
    alpha: float = 1.2e-5

    def __post_init__(self) -> None:
        if not (math.isfinite(self.alpha) and self.alpha >= 0.0):
            raise ValueError(
                f"Material {self.name!r}: coeficiente de dilatación térmica α = {self.alpha} no válido; "
                "debe ser un número finito ≥ 0 [1/°C] (acero ≈ 1,2·10⁻⁵)."
            )

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
        # Al 6061-T6: α = 23,6·10⁻⁶ 1/°C (20–100 °C; ASM Handbook Vol. 2 / Aluminum Association)
        Material("Al 6061-T6", E=68_900.0, Sy=276.0, Su=310.0, nu=0.33, rho=2700.0, alpha=2.36e-5),
    )
}

DEFAULT_MATERIAL = "ASTM A36"


# --------------------------------------------------------------------------- #
# Secciones
# --------------------------------------------------------------------------- #


_WEAK_SUFFIX = " (eje débil)"


@dataclass(frozen=True, slots=True)
class Section:
    """Sección transversal; flexiona alrededor del eje `I` en el plano del pórtico.

    `c_top` y `c_bot` son las distancias del eje neutro a las fibras extremas (y local + / −);
    distintas en secciones asimétricas (T, canales en eje débil vía `weak_axis()`).

    Corte (agregado al final, AGENTS.md §5): τ_máx = |V|·Q_max / (I·t_web) en el eje neutro.
    `Q_max = 0` o `t_web = 0` significa "sin datos de corte" (p. ej. sección personalizada sin
    Q): la verificación toma τ = 0 y el resultado es el de flexión pura de siempre.

    Eje débil: `Iy`, `Wy` = Iy / máx(c_y1, c_y2), distancias a fibras extremas `c_y1` / `c_y2`
    (en canales: c_y1 = e al dorso del alma, c_y2 = b − e a las puntas de ala) y `Q_y`, `t_y`
    para el corte cuando la sección flexiona en eje débil. Si se da `Iy` sin `c_y1`/`c_y2`, se toma
    c_y1 = c_y2 = b/2 (sección simétrica); si se da `Iy` sin `Wy`, se calcula Wy = Iy / c_y,máx.
    """

    name: str
    family: str
    A: float  # Área [mm²]
    I: float  # Momento de inercia eje de flexión (fuerte) [mm⁴]  # noqa: E741
    c_top: float  # Distancia a fibra superior [mm]
    c_bot: float  # Distancia a fibra inferior [mm]
    h: float  # Altura total [mm]
    b: float = 0.0  # Ancho [mm] (informativo)
    # ---- Corte y eje débil — agregados al final (AGENTS.md §5) ---- #
    Q_max: float = 0.0  # Momento estático de media sección respecto del eje neutro [mm³]
    t_web: float = 0.0  # Espesor resistente al corte en el eje neutro (alma; 2t en tubos) [mm]
    Iy: float = 0.0  # Momento de inercia eje débil [mm⁴]
    Wy: float = 0.0  # Módulo resistente elástico mínimo eje débil [mm³]
    c_y1: float = 0.0  # Eje débil: distancia a la fibra extrema del lado 1 (dorso del alma) [mm]
    c_y2: float = 0.0  # Eje débil: distancia a la fibra extrema del lado 2 (puntas de ala) [mm]
    Q_y: float = 0.0  # Momento estático para corte en flexión de eje débil [mm³]
    t_y: float = 0.0  # Espesor resistente al corte en flexión de eje débil [mm]
    # ---- Estabilidad (pandeo flexional / lateral-torsional) — agregados al final (AGENTS.md §5) ---- #
    Z: float = 0.0  # Módulo plástico respecto del eje de flexión [mm³]; 0 = sin dato (se usa W)
    Z_y: float = 0.0  # Módulo plástico respecto del eje débil [mm³]; 0 = sin dato
    J: float = 0.0  # Constante de torsión de Saint-Venant [mm⁴]; 0 = sin dato
    Cw: float = 0.0  # Constante de alabeo [mm⁶]; 0 = sin dato (sólo doble T y canales)
    h0: float = 0.0  # Distancia entre baricentros de alas [mm]; 0 = sin dato (sólo doble T y canales)

    def __post_init__(self) -> None:
        extra = (self.Q_max, self.t_web, self.Iy, self.Wy, self.c_y1, self.c_y2, self.Q_y, self.t_y,
                 self.Z, self.Z_y, self.J, self.Cw, self.h0)
        if not all(v >= 0.0 for v in extra):  # también rechaza NaN
            raise ValueError(
                f"Sección {self.name!r}: Q, t, Iy, Wy, Z, J, Cw, h0 y distancias de eje débil deben ser ≥ 0."
            )
        if self.Iy > 0.0 and self.c_y1 == 0.0 and self.c_y2 == 0.0 and self.b > 0.0:
            object.__setattr__(self, "c_y1", self.b / 2.0)
            object.__setattr__(self, "c_y2", self.b / 2.0)
        if self.Iy > 0.0 and self.Wy == 0.0 and self.c_y_max > 0.0:
            object.__setattr__(self, "Wy", self.Iy / self.c_y_max)
        # Módulo plástico de secciones simétricas respecto del eje: Z = 2·Q (media sección), exacto.
        if self.Z == 0.0 and self.Q_max > 0.0 and math.isclose(self.c_top, self.c_bot):
            object.__setattr__(self, "Z", 2.0 * self.Q_max)
        if self.Z_y == 0.0 and self.Q_y > 0.0 and self.c_y1 > 0.0 and math.isclose(self.c_y1, self.c_y2):
            object.__setattr__(self, "Z_y", 2.0 * self.Q_y)

    @property
    def c_max(self) -> float:
        return max(self.c_top, self.c_bot)

    @property
    def W(self) -> float:
        """Módulo resistente elástico mínimo [mm³] = I / c_max."""
        return self.I / self.c_max

    @property
    def W_top(self) -> float:
        """Módulo resistente de la fibra superior [mm³] = I / c_top."""
        return self.I / self.c_top

    @property
    def W_bot(self) -> float:
        """Módulo resistente de la fibra inferior [mm³] = I / c_bot."""
        return self.I / self.c_bot

    @property
    def c_y_max(self) -> float:
        """Distancia máxima a fibra extrema en eje débil [mm]."""
        return max(self.c_y1, self.c_y2)

    @property
    def has_shear_data(self) -> bool:
        """Hay Q_max y t_web para calcular τ = V·Q/(I·t)."""
        return self.Q_max > 0.0 and self.t_web > 0.0

    @property
    def shear_factor(self) -> float:
        """Q_max / (I·t_web) [1/mm²]: τ_máx = |V|·shear_factor. 0 si no hay datos de corte."""
        return self.Q_max / (self.I * self.t_web) if self.has_shear_data else 0.0

    # ---- Estabilidad ---- #

    @property
    def r_x(self) -> float:
        """Radio de giro respecto del eje de flexión (el de `I`) r_x = √(I/A) [mm]."""
        return math.sqrt(self.I / self.A) if self.A > 0.0 and self.I > 0.0 else 0.0

    @property
    def r_y(self) -> float:
        """Radio de giro respecto del otro eje principal (el de `Iy`) r_y = √(Iy/A) [mm]; 0 sin Iy."""
        return math.sqrt(self.Iy / self.A) if self.A > 0.0 and self.Iy > 0.0 else 0.0

    @property
    def r_min(self) -> float:
        """Radio de giro mínimo r_min = mín(r_x, r_y) [mm] (gobierna el pandeo flexional).

        Sin Iy (p. ej. `custom` sin Iy) sólo se conoce r_x: se devuelve r_x (pandeo en el plano).
        """
        return min((r for r in (self.r_x, self.r_y) if r > 0.0), default=0.0)

    @property
    def has_ltb_data(self) -> bool:
        """Hay Iy, J, Cw y h0 para el pandeo lateral-torsional de doble T y canales (AISC 360 F2)."""
        return self.Iy > 0.0 and self.J > 0.0 and self.Cw > 0.0 and self.h0 > 0.0

    @property
    def r_ts(self) -> float:
        """Radio de giro efectivo para PLT r_ts = √(√(Iy·Cw) / S) [mm] (AISC 360-16 ec. F2-7).

        En doble T simétricos equivale a r_ts² = Iy·h0 / (2·S). 0 si faltan Iy o Cw.
        """
        if not (self.Iy > 0.0 and self.Cw > 0.0):
            return 0.0
        return math.sqrt(math.sqrt(self.Iy * self.Cw) / self.W)

    def weak_axis(self, flip: bool = False) -> Section:
        """Misma sección orientada para flexionar en su eje débil dentro del plano del pórtico.

        Intercambia los ejes: I ↔ Iy, (c_top, c_bot) ↔ (c_y1, c_y2), Q_max ↔ Q_y, t_web ↔ t_y,
        h ↔ b, Z ↔ Z_y (J, Cw y h0 no dependen del eje; en eje débil no hay PLT, I < Iy). Por defecto la
        fibra superior es el lado 1 (dorso del alma en canales); `flip=True`
        invierte la sección (puntas de ala arriba). `s.weak_axis().weak_axis()` devuelve `s`.

        Raises:
            ValueError: la sección no tiene propiedades de eje débil (Iy, c_y1, c_y2).
        """
        if not (self.Iy > 0.0 and self.c_y1 > 0.0 and self.c_y2 > 0.0):
            raise ValueError(
                f"La sección {self.name!r} no tiene propiedades de eje débil (Iy, c_y1, c_y2)."
            )
        c_top, c_bot = (self.c_y2, self.c_y1) if flip else (self.c_y1, self.c_y2)
        name = (self.name.removesuffix(_WEAK_SUFFIX) if self.name.endswith(_WEAK_SUFFIX)
                else self.name + _WEAK_SUFFIX)
        return Section(
            name=name, family=self.family, A=self.A, I=self.Iy,
            c_top=c_top, c_bot=c_bot, h=self.b if self.b > 0.0 else self.c_y1 + self.c_y2, b=self.h,
            Q_max=self.Q_y, t_web=self.t_y, Iy=self.I, Wy=self.W,
            c_y1=self.c_top, c_y2=self.c_bot, Q_y=self.Q_max, t_y=self.t_web,
            Z=self.Z_y, Z_y=self.Z, J=self.J, Cw=self.Cw, h0=self.h0,
        )

    @property
    def mass_per_m(self) -> float:
        """Masa lineal de catálogo, ACERO (ρ = 7850 kg/m³) [kg/m]. Para otro material: `linear_mass`."""
        return self.linear_mass(7850.0)

    def linear_mass(self, rho: float) -> float:
        """Masa lineal [kg/m] = A [mm²] · 1e-6 [m²/mm²] · ρ [kg/m³] (cambio de unidad de la fórmula)."""
        return self.A * 1e-6 * rho


# ---- Constructores ---------------------------------------------------------- #


# Acuerdo alma-ala (enjuta = cuadrado r×r menos cuarto de círculo): área (1 − π/4)·r²,
# baricentro a r·(10 − 3π)/(12 − 3π) ≈ 0,2234·r de cada cara recta (Roark, tabla A.1).
_FILLET_AREA = 1.0 - math.pi / 4.0
_FILLET_CENTROID = (10.0 - 3.0 * math.pi) / (12.0 - 3.0 * math.pi)


def _i_shape_shear_weak(
    h: float, b: float, tw: float, tf: float, A: float, fillets: bool
) -> tuple[float, float]:
    """(Q_max [mm³], Iz eje débil [mm⁴]) de un doble T de alas paralelas por geometría.

    Rectángulos: Q = b·tf·(h − tf)/2 + tw·(h/2 − tf)²/2 ;  Iz = 2·tf·b³/12 + (h − 2tf)·tw³/12.
    Con `fillets`, se suman los 4 acuerdos de radio r, deducido del área de tabla A
    (4·(1 − π/4)·r² = A − A_rect); inercia propia de cada enjuta despreciada (≈ 0,0075·r⁴).
    """
    hw = h - 2.0 * tf
    Q = b * tf * (h - tf) / 2.0 + tw * (hw / 2.0) ** 2 / 2.0
    Iz = 2.0 * tf * b**3 / 12.0 + hw * tw**3 / 12.0
    a_r = (A - (2.0 * b * tf + hw * tw)) / 4.0  # área de cada acuerdo
    if fillets and a_r > 0.0:
        e = _FILLET_CENTROID * math.sqrt(a_r / _FILLET_AREA)
        Q += 2.0 * a_r * (h / 2.0 - tf - e)  # los dos acuerdos de la mitad superior
        Iz += 4.0 * a_r * (tw / 2.0 + e) ** 2
    return Q, Iz


def _i_shape_torsion(
    h: float, b: float, tw: float, tf: float, A: float, Iz: float, fillets: bool
) -> tuple[float, float, float]:
    """(J [mm⁴], Cw [mm⁶], h0 [mm]) de un doble T simétrico.

    J = 2/3·b·tf³ + 1/3·(h − 2tf)·tw³ [+ 2·α·D⁴ − 0,420·tf⁴ con acuerdos de radio r] (El Darwish y
    Johnston 1965; AISC Design Guide 9 §3.2), α = −0,042 + 0,2204·tw/tf + 0,1355·r/tf
    − 0,0865·r·tw/tf² − 0,0725·tw²/tf², D = [(tf + r)² + tw·(r + tw/4)] / (2r + tf); r deducido del área
    de tabla como en `_i_shape_shear_weak`. Cw = Iz·h0²/4 con h0 = h − tf (AISC 360-16, nota F2).
    """
    h0 = h - tf
    hw = h - 2.0 * tf
    J = 2.0 * b * tf**3 / 3.0 + hw * tw**3 / 3.0
    a_r = (A - (2.0 * b * tf + hw * tw)) / 4.0
    if fillets and a_r > 0.0:
        r = math.sqrt(a_r / _FILLET_AREA)
        alpha = (-0.042 + 0.2204 * tw / tf + 0.1355 * r / tf - 0.0865 * r * tw / tf**2
                 - 0.0725 * tw**2 / tf**2)
        D = ((tf + r) ** 2 + tw * (r + tw / 4.0)) / (2.0 * r + tf)
        J += 2.0 * alpha * D**4 - 0.420 * tf**4
    return J, Iz * h0**2 / 4.0, h0


def _channel_torsion(h: float, b: float, tw: float, tf: float) -> tuple[float, float, float]:
    """(J [mm⁴], Cw [mm⁶], h0 [mm]) de un canal de pared delgada (AISC Design Guide 9 §3.2).

    b' = b − tw/2, h0 = h − tf; J = (2·b'·tf³ + h0·tw³)/3;
    Cw = tf·b'³·h0²/12 · (3·b'·tf + 2·h0·tw) / (6·b'·tf + h0·tw).
    """
    bp = b - tw / 2.0
    h0 = h - tf
    J = (2.0 * bp * tf**3 + h0 * tw**3) / 3.0
    Cw = tf * bp**3 * h0**2 / 12.0 * (3.0 * bp * tf + 2.0 * h0 * tw) / (6.0 * bp * tf + h0 * tw)
    return J, Cw, h0


def rolled_i(
    name: str, family: str, h: float, b: float, A_cm2: float, I_cm4: float,
    tw: float = 0.0, tf: float = 0.0, Iz_cm4: float | None = None,
) -> Section:
    """Perfil laminado doble T simétrico a partir de valores de tabla (cm², cm⁴; tw, tf en mm).

    Corte: Q_max por geometría, t_web = tw. Eje débil: Iy = `Iz_cm4` de tabla si se da (alas
    inclinadas, IPN); si no, por geometría con acuerdos (alas paralelas, IPE). c_y = b/2;
    corte en eje débil lo toman las dos alas: Q_y = tf·b²/4, t_y = 2·tf.
    Estabilidad: J, Cw, h0 con `_i_shape_torsion` (acuerdos sólo en alas paralelas).
    Sin tw/tf (compatibilidad): sólo flexión en eje fuerte, sin datos de corte, eje débil ni PLT.
    """
    A = A_cm2 * 100.0
    Q = Iz = J = Cw = h0 = 0.0
    if tw > 0.0 and tf > 0.0:
        Q, Iz = _i_shape_shear_weak(h, b, tw, tf, A, fillets=Iz_cm4 is None)
        if Iz_cm4 is not None:
            Iz = Iz_cm4 * 1e4
        J, Cw, h0 = _i_shape_torsion(h, b, tw, tf, A, Iz, fillets=Iz_cm4 is None)
    return Section(
        name=name, family=family, A=A, I=I_cm4 * 1e4,
        c_top=h / 2.0, c_bot=h / 2.0, h=h, b=b,
        Q_max=Q, t_web=tw, Iy=Iz, Q_y=tf * b**2 / 4.0, t_y=2.0 * tf,
        J=J, Cw=Cw, h0=h0,
    )


def rolled_channel(
    name: str, h: float, b: float, A_cm2: float, I_cm4: float,
    tw: float = 0.0, tf: float = 0.0, Iz_cm4: float = 0.0, e_cm: float = 0.0,
) -> Section:
    """Perfil laminado U (UPN) a partir de valores de tabla (cm², cm⁴, cm; tw, tf en mm).

    Eje fuerte x-x: simétrico, c_top = c_bot = h/2; Q_max = b·tf·(h − tf)/2 + tw·(h/2 − tf)²/2
    con tf medio (≈ S_x de tabla), t_web = tw.
    Eje débil (asimétrico): Iy = `Iz_cm4` y e = `e_cm` de tabla (baricentro al dorso del alma):
    c_y1 = e, c_y2 = b − e, Wy = Iy/(b − e). Corte en eje débil por las dos alas cortadas en el eje
    neutro: Q_y = 2·tf·(b − e)²/2 = tf·(b − e)², t_y = 2·tf (alas de espesor medio).
    Estabilidad: J, Cw, h0 de pared delgada con tf medio (`_channel_torsion`).
    """
    Q = b * tf * (h - tf) / 2.0 + tw * (h / 2.0 - tf) ** 2 / 2.0 if tw > 0.0 and tf > 0.0 else 0.0
    e = e_cm * 10.0
    has_weak = Iz_cm4 > 0.0 and 0.0 < e < b
    J, Cw, h0 = _channel_torsion(h, b, tw, tf) if has_weak and tw > 0.0 and tf > 0.0 else (0.0, 0.0, 0.0)
    return Section(
        name=name, family="UPN", A=A_cm2 * 100.0, I=I_cm4 * 1e4,
        c_top=h / 2.0, c_bot=h / 2.0, h=h, b=b,
        Q_max=Q, t_web=tw, Iy=Iz_cm4 * 1e4 if has_weak else 0.0,
        c_y1=e if has_weak else 0.0, c_y2=b - e if has_weak else 0.0,
        Q_y=tf * (b - e) ** 2 if has_weak else 0.0, t_y=2.0 * tf if has_weak else 0.0,
        J=J, Cw=Cw, h0=h0,
    )


def wide_flange(
    name: str, d: float, bf: float, A: float, I: float,  # noqa: E741
    tw: float = 0.0, tf: float = 0.0,
) -> Section:
    """Perfil W (ala ancha, ASTM A6) a partir de valores de tabla ya en mm, mm², mm⁴.

    `d` es la altura real del perfil (no la nominal de la designación). Con tw y tf [mm]: Q_max e
    Iy por geometría con acuerdos (ver `_i_shape_shear_weak`), t_web = tw, Q_y = tf·bf²/4, t_y = 2tf.
    Estabilidad: J, Cw, h0 con acuerdos (`_i_shape_torsion`).
    """
    Q = Iz = J = Cw = h0 = 0.0
    if tw > 0.0 and tf > 0.0:
        Q, Iz = _i_shape_shear_weak(d, bf, tw, tf, A, fillets=True)
        J, Cw, h0 = _i_shape_torsion(d, bf, tw, tf, A, Iz, fillets=True)
    return Section(
        name=name, family="Perfil W", A=A, I=I,
        c_top=d / 2.0, c_bot=d / 2.0, h=d, b=bf,
        Q_max=Q, t_web=tw, Iy=Iz, Q_y=tf * bf**2 / 4.0, t_y=2.0 * tf,
        J=J, Cw=Cw, h0=h0,
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
    A = A_web + 2.0 * (A_fl + A_c)
    # Corte en eje fuerte: media sección = medio alma recto + ala + esquina superior
    Q = t * (hw / 2.0) ** 2 / 2.0 + A_fl * y_f + (A_c * d + Q_c0)
    # Eje débil (x desde la cara exterior del dorso del alma). Esquina: centro de curvatura en
    # x = r_o, la corona se extiende hacia el dorso; ∫x dA = r_o·A_c − Q_c0,
    # ∫x² dA = r_o²·A_c − 2·r_o·Q_c0 + I_c0. Alma recta: x = t/2, ∫x² = hw·t³/3.
    # Ala recta (x de r_o a B): ∫x dA = t·(B² − r_o²)/2, ∫x² dA = t·(B³ − r_o³)/3.
    Sx = A_web * t / 2.0 + 2.0 * t * (B**2 - r_o**2) / 2.0 + 2.0 * (r_o * A_c - Q_c0)
    Ixx = hw * t**3 / 3.0 + 2.0 * t * (B**3 - r_o**3) / 3.0 + 2.0 * (r_o**2 * A_c - 2.0 * r_o * Q_c0 + I_c0)
    e = Sx / A  # baricentro al dorso del alma
    # Q_y: tramos rectos de las dos alas más allá del eje neutro (= t·(B − e)² si e ≥ r_o; con alas
    # muy cortas, e < r_o, se desprecia la porción de esquina pasado el eje: aproximación).
    x_cut = max(e, r_o)
    Q_y = 2.0 * t * (B - x_cut) * ((B + x_cut) / 2.0 - e)
    # Estabilidad: modelo de línea media de pared delgada con tf = tw = t (esquinas rectas).
    J, Cw, h0 = _channel_torsion(H, B, t, t)
    return Section(
        name=name, family="Perfil C (Conformado)",
        A=A, I=I_web + 2.0 * (I_fl + I_c),
        c_top=H / 2.0, c_bot=H / 2.0, h=H, b=B,
        Q_max=Q, t_web=t, Iy=Ixx - A * e**2, c_y1=e, c_y2=B - e, Q_y=Q_y, t_y=2.0 * t,
        J=J, Cw=Cw, h0=h0,
    )


def chs(name: str, D: float, t: float) -> Section:
    """Tubo circular (CHS / caño).

    Corte: media corona Q = (D³ − d³)/12, dos paredes t_web = 2t (τ_máx ≈ 2V/A en pared delgada).
    Simetría polar: eje débil = eje fuerte.
    """
    d = D - 2.0 * t
    I = math.pi / 64.0 * (D**4 - d**4)  # noqa: E741
    Q = (D**3 - d**3) / 12.0
    return Section(
        name=name, family="Tubo circular",
        A=math.pi / 4.0 * (D**2 - d**2), I=I,
        c_top=D / 2.0, c_bot=D / 2.0, h=D, b=D,
        Q_max=Q, t_web=2.0 * t, Iy=I, Q_y=Q, t_y=2.0 * t,
    )


def rhs(name: str, B: float, H: float, t: float) -> Section:
    """Tubo rectangular / cuadrado (esquinas vivas). H = altura en el plano de flexión.

    Corte: Q = [B·H² − (B − 2t)(H − 2t)²]/8, dos almas t_web = 2t. Eje débil: Iy, Q_y con B ↔ H.
    """
    family = "Tubo cuadrado" if math.isclose(B, H) else "Tubo rectangular"
    Bi, Hi = B - 2.0 * t, H - 2.0 * t
    return Section(
        name=name, family=family,
        A=B * H - Bi * Hi,
        I=(B * H**3 - Bi * Hi**3) / 12.0,
        c_top=H / 2.0, c_bot=H / 2.0, h=H, b=B,
        Q_max=(B * H**2 - Bi * Hi**2) / 8.0, t_web=2.0 * t,
        Iy=(H * B**3 - Hi * Bi**3) / 12.0, Q_y=(H * B**2 - Hi * Bi**2) / 8.0, t_y=2.0 * t,
    )


def solid_rect(name: str, b: float, h: float) -> Section:
    """Barra rectangular maciza / planchuela. Corte: Q = b·h²/8, t = b (τ_máx = 1,5·V/A)."""
    return Section(
        name=name, family="Macizo", A=b * h, I=b * h**3 / 12.0,
        c_top=h / 2.0, c_bot=h / 2.0, h=h, b=b,
        Q_max=b * h**2 / 8.0, t_web=b, Iy=h * b**3 / 12.0, Q_y=h * b**2 / 8.0, t_y=h,
    )


def solid_round(name: str, d: float) -> Section:
    """Barra redonda maciza. Corte: Q = d³/12, t = d (τ_máx = 4V/(3A) en el diámetro)."""
    I = math.pi * d**4 / 64.0  # noqa: E741
    return Section(
        name=name, family="Macizo", A=math.pi * d**2 / 4.0, I=I,
        c_top=d / 2.0, c_bot=d / 2.0, h=d, b=d,
        Q_max=d**3 / 12.0, t_web=d, Iy=I, Q_y=d**3 / 12.0, t_y=d,
    )


def custom(
    name: str, A: float, I: float, c: float,  # noqa: E741
    *, Q_max: float = 0.0, t_web: float = 0.0, Iy: float = 0.0, Wy: float = 0.0,
    Z: float = 0.0, J: float = 0.0, Cw: float = 0.0, h0: float = 0.0,
) -> Section:
    """Sección definida por el usuario (simétrica en el plano de flexión).

    Q_max y t_web opcionales: sin ellos no se calcula τ (verificación de flexión pura).
    Iy y Wy opcionales (sin distancias de eje débil no admite `weak_axis()`); Iy da r_y para pandeo.
    Estabilidad opcional: Z (si no, 2·Q_max o, sin Q, se usa W), y J, Cw, h0 para el PLT de un doble T o
    canal (AISC 360 F2); sin ellos el PLT no se evalúa.
    """
    return Section(
        name=name, family="Personalizado", A=A, I=I, c_top=c, c_bot=c, h=2.0 * c,
        Q_max=Q_max, t_web=t_web, Iy=Iy, Wy=Wy, Z=Z, J=J, Cw=Cw, h0=h0,
    )


# ---- Catálogo ---------------------------------------------------------------- #

# IPE — Euronorm 19-57 / EN 10365: (h [mm], b [mm], A [cm²], Ix [cm⁴], tw [mm], tf [mm]).
# Iz (eje débil) y Q se calculan de la geometría con acuerdos (ver `_i_shape_shear_weak`).
_IPE: dict[int, tuple[float, float, float, float, float, float]] = {
    80: (80, 46, 7.64, 80.1, 3.8, 5.2),
    100: (100, 55, 10.3, 171.0, 4.1, 5.7),
    120: (120, 64, 13.2, 318.0, 4.4, 6.3),
    140: (140, 73, 16.4, 541.0, 4.7, 6.9),
    160: (160, 82, 20.1, 869.0, 5.0, 7.4),
    180: (180, 91, 23.9, 1317.0, 5.3, 8.0),
    200: (200, 100, 28.5, 1943.0, 5.6, 8.5),
    220: (220, 110, 33.4, 2772.0, 5.9, 9.2),
    240: (240, 120, 39.1, 3892.0, 6.2, 9.8),
    270: (270, 135, 45.9, 5790.0, 6.6, 10.2),
    300: (300, 150, 53.8, 8356.0, 7.1, 10.7),
    330: (330, 160, 62.6, 11770.0, 7.5, 11.5),
    360: (360, 170, 72.7, 16270.0, 8.0, 12.7),
    400: (400, 180, 84.5, 23130.0, 8.6, 13.5),
    450: (450, 190, 98.8, 33740.0, 9.4, 14.6),
    500: (500, 200, 116.0, 48200.0, 10.2, 16.0),
    600: (600, 220, 156.0, 92080.0, 12.0, 19.0),
}

# IPN — DIN 1025-1 (alas inclinadas 14 %): (h [mm], b [mm], A [cm²], Ix [cm⁴], tw [mm],
# tf medio [mm], Iz [cm⁴]). Iz de tabla: la geometría con tf medio lo sobrestima (alas en cuña).
_IPN: dict[int, tuple[float, float, float, float, float, float, float]] = {
    80: (80, 42, 7.57, 77.8, 3.9, 5.9, 6.29),
    100: (100, 50, 10.6, 171.0, 4.5, 6.8, 12.2),
    120: (120, 58, 14.2, 328.0, 5.1, 7.7, 21.5),
    140: (140, 66, 18.2, 573.0, 5.7, 8.6, 35.2),
    160: (160, 74, 22.8, 935.0, 6.3, 9.5, 54.7),
    180: (180, 82, 27.9, 1450.0, 6.9, 10.4, 81.3),
    200: (200, 90, 33.4, 2140.0, 7.5, 11.3, 117.0),
    220: (220, 98, 39.5, 3060.0, 8.1, 12.2, 162.0),
    240: (240, 106, 46.1, 4250.0, 8.7, 13.1, 221.0),
    260: (260, 113, 53.3, 5740.0, 9.4, 14.1, 288.0),
    280: (280, 119, 61.0, 7590.0, 10.1, 15.2, 364.0),
    300: (300, 125, 69.0, 9800.0, 10.8, 16.2, 451.0),
}

# UPN — DIN 1026-1 (EN 10279): (h [mm], b [mm], A [cm²], Ix [cm⁴], tw [mm], tf medio [mm],
# Iy eje débil [cm⁴], e [cm] = distancia del baricentro al dorso del alma).
_UPN: dict[int, tuple[float, float, float, float, float, float, float, float]] = {
    80: (80, 45, 11.0, 106.0, 6.0, 8.0, 19.4, 1.45),
    100: (100, 50, 13.5, 206.0, 6.0, 8.5, 29.3, 1.55),
    120: (120, 55, 17.0, 364.0, 7.0, 9.0, 43.2, 1.60),
    140: (140, 60, 20.4, 605.0, 7.0, 10.0, 62.7, 1.75),
    160: (160, 65, 24.0, 925.0, 7.5, 10.5, 85.3, 1.84),
    180: (180, 70, 28.0, 1350.0, 8.0, 11.0, 114.0, 1.92),
    200: (200, 75, 32.2, 1910.0, 8.5, 11.5, 148.0, 2.01),
    220: (220, 80, 37.4, 2690.0, 9.0, 12.5, 197.0, 2.14),
    240: (240, 85, 42.3, 3600.0, 9.5, 13.0, 248.0, 2.23),
    260: (260, 90, 48.3, 4820.0, 10.0, 14.0, 317.0, 2.36),
    280: (280, 95, 53.3, 6280.0, 10.0, 15.0, 399.0, 2.53),
    300: (300, 100, 58.8, 8030.0, 10.0, 16.0, 495.0, 2.70),
}

# Perfiles W — ASTM A6/A6M, AISC Shapes Database v15.0. Valores de la tabla en
# unidades US convertidos a SI (1 in = 25,4 mm) y redondeados; equivalente US en
# el comentario. (d [mm], bf [mm], A [mm²], Ix [mm⁴], tw [mm], tf [mm]); d = altura real.
_W: dict[str, tuple[float, float, float, float, float, float]] = {
    "W 150x13": (148.1, 100.1, 1626.0, 6.202e6, 4.32, 4.95),  # W6x8.5  (tw 0.170 in, tf 0.195 in)
    "W 150x18": (153.2, 101.6, 2290.0, 9.199e6, 5.84, 7.11),  # W6x12   (tw 0.230 in, tf 0.280 in)
    "W 200x15": (200.4, 100.1, 1910.0, 12.82e6, 4.32, 5.21),  # W8x10   (tw 0.170 in, tf 0.205 in)
    "W 200x22.5": (206.0, 102.0, 2865.0, 19.98e6, 6.22, 8.00),  # W8x15  (tw 0.245 in, tf 0.315 in)
    "W 250x28.4": (260.1, 102.1, 3626.0, 40.08e6, 6.35, 10.03),  # W10x19 (tw 0.250 in, tf 0.395 in)
    "W 310x38.7": (310.4, 164.8, 4935.0, 84.91e6, 5.84, 9.65),  # W12x26 (tw 0.230 in, tf 0.380 in)
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

# Tubos estructurales: (H, B, t) en mm. H = lado mayor = altura en el plano de flexión (eje fuerte),
# B = ancho (eje débil), igual que FAMILY_NOTES["Tubo rectangular"]; designación "Tubo H×B×t".
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

    for n, (h, b, A, I, tw, tf) in _IPE.items():
        add(rolled_i(f"IPE {n}", "IPE", h, b, A, I, tw, tf))
    for n, (h, b, A, I, tw, tf, Iz) in _IPN.items():
        add(rolled_i(f"IPN {n}", "IPN", h, b, A, I, tw, tf, Iz_cm4=Iz))
    for n, (h, b, A, I, tw, tf, Iz, e) in _UPN.items():
        add(rolled_channel(f"UPN {n}", h, b, A, I, tw, tf, Iz, e))
    for name, (d, bf, A, I, tw, tf) in _W.items():
        add(wide_flange(name, d, bf, A, I, tw, tf))
    for H, B, t in _C_COLD:
        add(cold_formed_channel(f"C {H:g}x{B:g}x{t:g}", H, B, t))
    for nps, D, t in _PIPE_SCH40:
        add(chs(f"Caño {nps} Sch40 (Ø{D:g}x{t:g})", D, t))
    for H, B, t in _RHS:
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

CLOSED_FAMILIES: frozenset[str] = frozenset({"Tubo circular", "Tubo cuadrado", "Tubo rectangular"})
"""Secciones cerradas: el PLT no gobierna en proporciones usuales (AISC 360-16 F7, F8)."""

SOLID_FAMILY = "Macizo"
"""Barras macizas: rectangulares de canto con PLT según AISC 360-16 F11; redondas sin PLT."""

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
