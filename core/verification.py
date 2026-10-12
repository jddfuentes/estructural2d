"""Verificación estructural (postprocesamiento). Unidades: MPa, mm.

Criterios MVP:
- Resistencia: tensión elástica equivalente de von Mises σ_vM (normal N/A ± M·c/I en la fibra
  extrema y corte en el eje neutro, ver abajo) frente a la fluencia del material de cada barra.
  FS = Sy / σ_vM ≥ FS_min (= Sy / σ_max cuando gobierna la flexión, caso usual).
- Servicio (ELS): flecha relativa L/δ_max ≥ N (por defecto L/300), con δ_max el
  desplazamiento total máximo de la estructura y L la longitud de referencia
  (por defecto, el ancho horizontal del modelo = luz de la viga o del pórtico).

Combinaciones de carga (`check_combinations`, hipótesis D y L):
- ELU (resistencia), envolvente de 1.4D y 1.2D + 1.6L (LRFD): σ_u ≤ φ·Sy con φ = 0.90, sobre la
  tensión elástica equivalente σ_vM (= normal en flexión dominante) de las solicitaciones
  mayoradas (criterio de primera fluencia, conservador frente al momento plástico). Gobierna la
  combinación de máxima utilización σ_u / (φ·Sy), que es el máximo de la envolvente.
- ELS (servicio): flecha bajo 1.0D + 1.0L (cargas sin mayorar) frente a L/N.

Corte y tensión equivalente (`member_stresses`, AGENTS.md §4):
- τ(x) = |V(x)|·Q_max / (I·t_web), máxima en el eje neutro (Jourawski, elástico).
- Tensiones normales con fibras asimétricas (convención del solver: M > 0 tracciona la inferior):
  σ_top = N/A − M·c_top/I,  σ_bot = N/A + M·c_bot/I,  σ_max = máx(|σ_top|, |σ_bot|).
- von Mises en dos puntos de la sección: fibra extrema (τ ≈ 0) → σ_vM,flex = σ_max; eje neutro
  (σ = N/A, τ = τ_máx) → σ_vM,corte = √((N/A)² + 3·τ²). Gobierna σ_vM = máx de ambos.
  FS = Sy / σ_vM,máx (y en ELU σ_vM,u ≤ φ·Sy). En flexión dominante σ_vM = σ_max y el resultado es
  idéntico al de tensión normal pura; una sección sin datos de corte (Q = 0) toma τ = 0.
- Cargas térmicas / deformaciones impuestas: el N de `MemberResult` ya incluye la fuerza de fijación
  −E·A·ε₀ (ver `core.solver`), así que σ = N/A ± M·c/I, σ_vM, ELU y la compresión N_comp de la
  interacción H1 (estabilidad) incorporan los esfuerzos térmicos sin tratamiento aparte.
- Eje débil: la sección se orienta con `Section.weak_axis()` (análisis plano en eje débil) o, para
  un momento fuera del plano M_y, `weak_axis_bending_stress` da σ = |M_y| / W_y.

Estabilidad (opcional, `stability=True` en `check_safety` / `check_combinations`; AISC 360-16 =
CIRSOC 301-2005, misma formulación LRFD):
- Barra física (`physical_members`): cadena de barras colineales consecutivas del modelo con igual sección
  y material, cortada en apoyos, resortes y nudos con ≠ 2 barras. Los constructores parten una viga en cada
  carga puntual: para el pandeo cuenta la barra entera, no el tramo entre cargas.
- Pandeo flexional (`column_buckling_capacity`, cap. E3): λ = K·L/r_min, Fe = π²E/λ²;
  λ ≤ 4,71·√(E/Fy) → Fcr = 0,658^(Fy/Fe)·Fy; si no → Fcr = 0,877·Fe; Pn = Fcr·A.
- Pandeo lateral-torsional (`beam_ltb_capacity`): doble T y canales por §F2 (Lp, Lr, Fcr con J, Cw, r_ts),
  barras rectangulares macizas de canto por §F11; Mp = Fy·Z ≤ 1,6·Fy·S. Sin PLT (Mn = Mp): flexión en eje
  débil, secciones cerradas (§F7, F8) y redondos macizos.
- Interacción flexocompresión (`interaction_h1`, §H1.1) con la compresión máxima y el |M| máximo de la
  barra física: U = Pr/Pc + 8/9·Mr/Mc si Pr/Pc ≥ 0,2; U = Pr/(2Pc) + Mr/Mc si no. Pc = Pn/FS_min y
  Mc = Mn/FS_min en `check_safety` (en ELU, FS_min = 1/φ ⇒ Pc = φc·Pn, Mc = φb·Mn).
- Supuestos (predimensionamiento): Cb = 1 (conservador); KL con la longitud de la barra física y r_min,
  sin aprovechar arriostramientos intermedios (conservador); Lb = mín(`unbraced_length`, L). Solicitaciones
  de primer orden (sin amplificación B1/B2 ni análisis directo §C): en pórticos desplazables el K elegido
  debe representar el desplazamiento lateral. Sin pandeo local (secciones esbeltas, §E7), torsional ni
  flexotorsional (§E4). Fórmulas de acero: para aluminio u otros materiales son sólo orientativas.

No incluye abolladura local, tensión combinada en la unión ala-alma de perfiles doble T (se evalúan sólo
fibra extrema y eje neutro), flexión esviada combinada ni fatiga.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.materials import CLOSED_FAMILIES, SOLID_FAMILY, Section
from core.model import COMB_SERVICE_D_L, SLS, ULS, ULS_COMBINATIONS, LoadCase, LoadCombination, Model
from core.solver import FloatArray, MemberResult, Results, combine_cases, solve_combinations

FS_MIN_DEFAULT = 1.5

# mm -> m, N -> kN y N·mm -> kN·m SÓLO para redactar `issues` (texto listo para mostrar). Excepción de
# presentación documentada en AGENTS.md §4: ningún número convertido sale de este módulo.
_MM_TO_M_TEXT = 1e-3
_N_TO_KN_TEXT = 1e-3
_NMM_TO_KNM_TEXT = 1e-6


def _where(member: int, point: tuple[float, float]) -> str:
    """Ubicación para los mensajes de `issues`: 'barra k, (x; y) m'."""
    return f"barra {member}, ({point[0] * _MM_TO_M_TEXT:.2f}; {point[1] * _MM_TO_M_TEXT:.2f}) m"

# Flecha admisible por defecto L/300: valor usual de predimensionamiento para vigas.
# Los límites de servicio dependen del uso y de la norma (AISC 360 cap. L y CIRSOC 301
# remiten a criterios de servicio por tipo de elemento): ajustar con `deflection_limit_ratio`.
DEFLECTION_LIMIT_DEFAULT = 300.0

# Factor de resistencia LRFD a flexión φb = 0.90: AISC 360-16 §F1 y CIRSOC 301-2005 §F.1.
PHI_DEFAULT = 0.90


# Tolerancia relativa para decidir que el corte (von Mises en el eje neutro) gobierna sobre la
# tensión normal: por debajo se considera empate y se informa como flexión (textos de siempre).
_SHEAR_GOVERNS_RTOL = 1e-12


def shear_stress(section: Section, V: FloatArray | float) -> FloatArray:
    """Tensión de corte máxima de la sección (eje neutro) τ = |V|·Q_max / (I·t_web) [MPa].

    V en N, Q en mm³, I en mm⁴, t en mm. Devuelve ceros si la sección no tiene datos de corte.
    """
    return np.abs(np.asarray(V, dtype=np.float64)) * section.shear_factor


def von_mises(sigma: FloatArray | float, tau: FloatArray | float) -> FloatArray:
    """Tensión equivalente de von Mises para estado plano σ-τ: σ_vM = √(σ² + 3·τ²) [MPa]."""
    s = np.asarray(sigma, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    out: FloatArray = np.sqrt(s * s + 3.0 * t * t)
    return out


def weak_axis_bending_stress(section: Section, M_y: float) -> float:
    """Tensión normal máxima por flexión en eje débil σ = |M_y| / W_y [MPa] (M_y en N·mm).

    Raises:
        ValueError: la sección no tiene módulo resistente de eje débil (Wy ≤ 0).
    """
    if not section.Wy > 0.0:
        raise ValueError(f"La sección {section.name!r} no tiene módulo resistente de eje débil Wy.")
    return abs(M_y) / section.Wy


@dataclass(frozen=True, slots=True)
class MemberStresses:
    """Tensiones de una barra en las estaciones de su `MemberResult` [MPa]."""

    member: int
    x: FloatArray  # estaciones locales [mm]
    sigma_top: FloatArray  # σ fibra superior = N/A − M·c_top/I
    sigma_bot: FloatArray  # σ fibra inferior = N/A + M·c_bot/I
    sigma_max: FloatArray  # máx(|σ_top|, |σ_bot|) (fibra extrema)
    sigma_axial: FloatArray  # N/A (eje neutro)
    tau: FloatArray  # τ máx de la sección = |V|·Q/(I·t) (eje neutro)
    sigma_vm_flex: FloatArray  # von Mises en la fibra extrema (τ ≈ 0) = σ_max
    sigma_vm_shear: FloatArray  # von Mises en el eje neutro = √((N/A)² + 3τ²)
    sigma_vm: FloatArray  # máx(σ_vM,flex, σ_vM,corte): tensión equivalente gobernante


def member_stresses(section: Section, mr: MemberResult) -> MemberStresses:
    """σ normales, τ de corte y von Mises en cada estación de la barra (ver docstring del módulo).

    Usa σ_top / σ_bot del solver (convención AGENTS.md §4, exactas con c_top ≠ c_bot). N incluye el
    esfuerzo axial de origen térmico (N = N_cinemático − E·A·α·ΔT), por lo que σ_axial = N/A también.
    """
    sigma_max = mr.sigma_abs
    sigma_axial = mr.N / section.A
    tau = shear_stress(section, mr.V)
    vm_shear = von_mises(sigma_axial, tau)
    return MemberStresses(
        member=mr.member, x=mr.x, sigma_top=mr.sigma_top, sigma_bot=mr.sigma_bot,
        sigma_max=sigma_max, sigma_axial=sigma_axial, tau=tau,
        sigma_vm_flex=sigma_max, sigma_vm_shear=vm_shear, sigma_vm=np.maximum(sigma_max, vm_shear),
    )


# --------------------------------------------------------------------------- #
# Estabilidad: pandeo flexional (AISC 360-16 cap. E), PLT (F2, F11) e interacción (H1).
# CIRSOC 301-2005 (caps. E, F, H) adopta la misma formulación LRFD.
# --------------------------------------------------------------------------- #

# Módulo de elasticidad del acero E = 200 000 MPa (AISC 360-16, Glosario / §E3; CIRSOC 301 §A.3).
E_STEEL = 200_000.0
# Factores de resistencia LRFD: compresión φc = 0,90 (AISC 360-16 §E1), flexión φb = 0,90 (§F1).
PHI_C_DEFAULT = 0.90
PHI_B_DEFAULT = 0.90
# Esbeltez máxima recomendada de barras comprimidas KL/r ≤ 200: AISC 360-16 §E2 (nota de usuario).
SLENDERNESS_MAX = 200.0
# Cb = 1 corresponde a momento uniforme: conservador para cualquier diagrama (AISC 360-16 §F1).
CB_DEFAULT = 1.0
# Tope del momento plástico Mp ≤ 1,6·My (AISC 360-16 ecs. F6-1 y F11-1).
_MP_MAX_OVER_MY = 1.6


class BucklingRegime(str, Enum):
    INELASTIC = "inelástico"  # λ ≤ 4,71·√(E/Fy): Fcr = 0,658^(Fy/Fe)·Fy  (AISC 360-16 E3-2)
    ELASTIC = "elástico"  # λ > 4,71·√(E/Fy): Fcr = 0,877·Fe  (AISC 360-16 E3-3)


@dataclass(frozen=True, slots=True)
class ColumnBucklingCheck:
    """Pandeo flexional de una barra comprimida (AISC 360-16 §E3 / CIRSOC 301 §E.3). Unidades SI-mm."""

    section: str
    L: float  # longitud de la barra [mm]
    K: float  # factor de longitud efectiva [-]
    r_min: float  # radio de giro mínimo [mm]
    slenderness: float  # λ = K·L / r_min [-]
    slenderness_limit: float  # 4,71·√(E/Fy) [-]: transición inelástico / elástico
    Fe: float  # tensión de Euler π²E/λ² [MPa] (inf si λ = 0)
    Fcr: float  # tensión crítica de pandeo [MPa]
    Fy: float  # [MPa]
    E: float  # [MPa]
    A: float  # [mm²]
    regime: BucklingRegime
    Pn: float  # resistencia nominal Fcr·A [N]
    phi_c: float  # factor de resistencia [-]
    N_comp: float = 0.0  # compresión solicitante, valor absoluto [N]

    @property
    def KL(self) -> float:
        """Longitud efectiva K·L [mm]."""
        return self.K * self.L

    @property
    def phi_Pn(self) -> float:
        """Resistencia de diseño φc·Pn [N]."""
        return self.phi_c * self.Pn

    @property
    def utilization(self) -> float:
        """|N_comp| / (φc·Pn); > 1 no verifica."""
        return abs(self.N_comp) / self.phi_Pn

    @property
    def slenderness_ok(self) -> bool:
        """KL/r ≤ 200 (recomendación AISC 360-16 §E2)."""
        return self.slenderness <= SLENDERNESS_MAX


def _check_material_inputs(Fy: float, E: float, phi: float, what: str) -> None:
    if not (Fy > 0.0 and math.isfinite(Fy)):
        raise ValueError("La tensión de fluencia Fy debe ser positiva [MPa].")
    if not (E > 0.0 and math.isfinite(E)):
        raise ValueError("El módulo de elasticidad E debe ser positivo [MPa].")
    if not 0.0 < phi <= 1.0:  # también rechaza NaN
        raise ValueError(f"El factor de resistencia {what} debe estar en el intervalo (0; 1].")


def column_buckling_capacity(
    section: Section,
    L: float,
    K: float = 1.0,
    Fy: float = 250.0,
    phi_c: float = PHI_C_DEFAULT,
    *,
    E: float = E_STEEL,
    N_comp: float = 0.0,
) -> ColumnBucklingCheck:
    """Resistencia a compresión por pandeo flexional (AISC 360-16 §E3 / CIRSOC 301 §E.3).

    λ = K·L/r_min; Fe = π²E/λ²; λ ≤ 4,71·√(E/Fy) (⇔ Fy/Fe ≤ 2,25) → Fcr = 0,658^(Fy/Fe)·Fy, si no
    Fcr = 0,877·Fe; Pn = Fcr·A; φPn = φc·Pn. L = 0 da Fcr = Fy (aplastamiento).

    Args:
        L: longitud de la barra [mm]; K: factor de longitud efectiva; Fy, E [MPa].
        N_comp: compresión solicitante [N] (se toma |N_comp|) para `utilization`.

    Raises:
        ValueError: L < 0, K ≤ 0, Fy o E ≤ 0, φc fuera de (0; 1] o sección sin A / radio de giro.
    """
    if not (L >= 0.0 and math.isfinite(L)):
        raise ValueError("La longitud de la barra L debe ser ≥ 0 [mm].")
    if not (K > 0.0 and math.isfinite(K)):
        raise ValueError("El factor de longitud efectiva K debe ser positivo.")
    _check_material_inputs(Fy, E, phi_c, "φc")
    r = section.r_min
    if not (section.A > 0.0 and r > 0.0):
        raise ValueError(f"La sección {section.name!r} necesita área e inercia positivas para el pandeo.")
    lam = K * L / r
    lam_lim = 4.71 * math.sqrt(E / Fy)  # AISC 360-16 §E3(a)
    Fe = math.pi**2 * E / lam**2 if lam > 0.0 else math.inf  # E3-4
    if lam <= lam_lim:
        regime, Fcr = BucklingRegime.INELASTIC, 0.658 ** (Fy / Fe) * Fy  # E3-2
    else:
        regime, Fcr = BucklingRegime.ELASTIC, 0.877 * Fe  # E3-3
    return ColumnBucklingCheck(
        section=section.name, L=L, K=K, r_min=r, slenderness=lam, slenderness_limit=lam_lim,
        Fe=Fe, Fcr=Fcr, Fy=Fy, E=E, A=section.A, regime=regime, Pn=Fcr * section.A,
        phi_c=phi_c, N_comp=abs(N_comp),
    )


class LTBZone(str, Enum):
    PLASTIC = "plastificación"  # Lb ≤ Lp: Mn = Mp
    INELASTIC = "PLT inelástico"  # Lp < Lb ≤ Lr: interpolación lineal entre Mp y 0,7·Fy·S
    ELASTIC = "PLT elástico"  # Lb > Lr: Mn = Fcr·S
    NOT_APPLICABLE = "sin PLT"  # eje débil, sección cerrada o redondo macizo: Mn = Mp
    NOT_EVALUATED = "PLT no evaluado"  # faltan Iy, J, Cw o h0 (sección personalizada): Mn = Mp


@dataclass(frozen=True, slots=True)
class BeamLTBCheck:
    """Momento resistente con pandeo lateral-torsional (AISC 360-16 §F2 / §F11). Unidades SI-mm."""

    section: str
    Lb: float  # longitud no arriostrada lateralmente [mm]
    Cb: float  # factor de modificación por diagrama de momentos [-]
    Fy: float  # [MPa]
    E: float  # [MPa]
    S: float  # módulo resistente elástico [mm³]
    Mp: float  # momento plástico Fy·Z ≤ 1,6·Fy·S [N·mm] (sin Z: Fy·S)
    Lp: float  # límite de plastificación [mm] (inf si no hay PLT)
    Lr: float  # límite de PLT inelástico [mm] (inf si no hay PLT)
    Fcr: float  # tensión crítica elástica de PLT con Lb [MPa] (inf si no hay PLT o Lb = 0)
    Mn: float  # momento nominal [N·mm]
    phi_b: float  # factor de resistencia [-]
    zone: LTBZone
    method: str  # norma aplicada o motivo por el que no hay PLT (castellano)
    M_u: float = 0.0  # momento solicitante, valor absoluto [N·mm]

    @property
    def My(self) -> float:
        """Momento de primera fluencia Fy·S [N·mm]."""
        return self.Fy * self.S

    @property
    def phi_Mn(self) -> float:
        """Momento de diseño φb·Mn [N·mm]."""
        return self.phi_b * self.Mn

    @property
    def utilization(self) -> float:
        """|M_u| / (φb·Mn); > 1 no verifica."""
        return abs(self.M_u) / self.phi_Mn

    @property
    def reduced(self) -> bool:
        """El PLT reduce la resistencia por debajo de Mp."""
        return self.Mn < self.Mp * (1.0 - 1e-12)


def _ltb_method(section: Section) -> tuple[str, str]:
    """(clave, texto) del PLT aplicable: "F2", "F11", "none" (sin PLT) o "na" (sin datos)."""
    if section.Iy > 0.0 and section.I <= section.Iy * (1.0 + 1e-9):
        return "none", ("flexión alrededor del eje de menor inercia o sección sin eje débil: "
                        "sin PLT (AISC 360-16 §F6, F8, F11)")
    if section.family in CLOSED_FAMILIES:
        return "none", "sección cerrada: el PLT no gobierna en proporciones usuales (AISC 360-16 §F7)"
    if section.has_ltb_data:
        return "F2", "AISC 360-16 §F2 (doble T y canales)"
    if section.family == SOLID_FAMILY and section.b > 0.0 and section.Iy > 0.0:
        return "F11", "AISC 360-16 §F11 (barra rectangular maciza de canto)"
    return "na", ("faltan Iy, J, Cw o h0: PLT no evaluado (se toma Mn = Mp; "
                  "no conservador si la barra no está arriostrada lateralmente)")


def beam_ltb_capacity(
    section: Section,
    Lb: float,
    Fy: float = 250.0,
    Cb: float = CB_DEFAULT,
    phi_b: float = PHI_B_DEFAULT,
    *,
    E: float = E_STEEL,
    M_u: float = 0.0,
) -> BeamLTBCheck:
    """Momento nominal con pandeo lateral-torsional para la longitud no arriostrada Lb.

    Mp = Fy·Z ≤ 1,6·Fy·S (sin módulo plástico Z se toma Mp = My = Fy·S, conservador).
    Doble T y canales (AISC 360-16 §F2), con c = (h0/2)·√(Iy/Cw) (= 1 en doble T) y jc = J·c/(S·h0):
      Lp = 1,76·r_y·√(E/Fy)                                                         (F2-5)
      Lr = 1,95·r_ts·E/(0,7·Fy)·√(jc + √(jc² + 6,76·(0,7·Fy/E)²))                   (F2-6)
      Lb ≤ Lp: Mn = Mp;  Lp < Lb ≤ Lr: Mn = Cb·[Mp − (Mp − 0,7·Fy·S)·(Lb − Lp)/(Lr − Lp)] ≤ Mp  (F2-2)
      Lb > Lr: Mn = Fcr·S ≤ Mp, Fcr = Cb·π²E/(Lb/r_ts)²·√(1 + 0,078·jc·(Lb/r_ts)²)   (F2-3, F2-4)
    Barra rectangular maciza de canto d × t (§F11), con Lb·d/t²:
      ≤ 0,08·E/Fy: Mn = Mp;  ≤ 1,9·E/Fy: Mn = Cb·[1,52 − 0,274·(Lb·d/t²)·Fy/E]·My ≤ Mp  (F11-2);
      si no: Mn = Fcr·S ≤ Mp, Fcr = 1,9·E·Cb/(Lb·d/t²)                               (F11-3, F11-4)
    Sin PLT (Mn = Mp): flexión en eje débil, secciones cerradas, redondos macizos; sin datos: ídem
    con `zone = NOT_EVALUATED`.

    Args:
        Lb: longitud no arriostrada [mm] (0 = arriostramiento continuo). Cb ≥ 1 (1 = conservador).
        M_u: momento solicitante [N·mm] (se toma |M_u|) para `utilization`.

    Raises:
        ValueError: Lb < 0, Cb < 1, Fy o E ≤ 0, φb fuera de (0; 1] o sección sin módulo resistente.
    """
    if not (Lb >= 0.0 and math.isfinite(Lb)):
        raise ValueError("La longitud no arriostrada Lb debe ser ≥ 0 [mm].")
    if not (Cb >= 1.0 and math.isfinite(Cb)):
        raise ValueError("El factor de momento Cb debe ser ≥ 1 (AISC 360-16 §F1).")
    _check_material_inputs(Fy, E, phi_b, "φb")
    if not (section.I > 0.0 and section.c_max > 0.0):
        raise ValueError(f"La sección {section.name!r} necesita inercia y distancia a fibra extrema > 0.")
    S = section.W
    My = Fy * S
    Mp = min(Fy * (section.Z if section.Z > 0.0 else S), _MP_MAX_OVER_MY * My)
    Lp = Lr = Fcr = math.inf
    kind, method = _ltb_method(section)
    if kind == "F2":
        rts, h0 = section.r_ts, section.h0
        c = h0 / 2.0 * math.sqrt(section.Iy / section.Cw)  # F2-8b; = 1 en doble T simétrico (F2-8a)
        jc = section.J * c / (S * h0)
        Lp = 1.76 * section.r_y * math.sqrt(E / Fy)  # F2-5
        Lr = 1.95 * rts * E / (0.7 * Fy) * math.sqrt(jc + math.sqrt(jc**2 + 6.76 * (0.7 * Fy / E) ** 2))
        if Lb > 0.0:
            s2 = (Lb / rts) ** 2
            Fcr = Cb * math.pi**2 * E / s2 * math.sqrt(1.0 + 0.078 * jc * s2)  # F2-4
        if Lb <= Lp:
            zone, Mn = LTBZone.PLASTIC, Mp
        elif Lb <= Lr:
            zone = LTBZone.INELASTIC
            Mn = min(Cb * (Mp - (Mp - 0.7 * Fy * S) * (Lb - Lp) / (Lr - Lp)), Mp)  # F2-2
        else:
            zone, Mn = LTBZone.ELASTIC, min(Fcr * S, Mp)  # F2-3
    elif kind == "F11":
        k = section.h / section.b**2  # d/t² [1/mm]
        Lp, Lr = 0.08 * E / (Fy * k), 1.9 * E / (Fy * k)
        if Lb > 0.0:
            Fcr = 1.9 * E * Cb / (Lb * k)  # F11-4
        if Lb <= Lp:
            zone, Mn = LTBZone.PLASTIC, Mp  # F11-1
        elif Lb <= Lr:
            zone, Mn = LTBZone.INELASTIC, min(Cb * (1.52 - 0.274 * Lb * k * Fy / E) * My, Mp)  # F11-2
        else:
            zone, Mn = LTBZone.ELASTIC, min(Fcr * S, Mp)  # F11-3
    else:
        zone = LTBZone.NOT_APPLICABLE if kind == "none" else LTBZone.NOT_EVALUATED
        Mn = Mp
    return BeamLTBCheck(
        section=section.name, Lb=Lb, Cb=Cb, Fy=Fy, E=E, S=S, Mp=Mp, Lp=Lp, Lr=Lr, Fcr=Fcr, Mn=Mn,
        phi_b=phi_b, zone=zone, method=method, M_u=abs(M_u),
    )


def interaction_h1(Pr: float, Pc: float, Mr: float, Mc: float) -> float:
    """Interacción flexocompresión AISC 360-16 §H1.1 (CIRSOC 301 §H.1.1); > 1 no verifica.

    Pr/Pc ≥ 0,2 → Pr/Pc + 8/9·Mr/Mc (H1-1a);  Pr/Pc < 0,2 → Pr/(2·Pc) + Mr/Mc (H1-1b).
    Pr, Mr en valor absoluto; Pc, Mc > 0 son las resistencias disponibles.
    """
    p, m = abs(Pr) / Pc, abs(Mr) / Mc
    return p + 8.0 / 9.0 * m if p >= 0.2 else p / 2.0 + m


def _unit_away(model: Model, member: int, node: int) -> tuple[float, float]:
    """Versor de la barra desde `node` hacia su otro extremo."""
    mem = model.members[member]
    other = mem.j if mem.i == node else mem.i
    (x0, y0), (x1, y1) = model.nodes[node], model.nodes[other]
    L = math.hypot(x1 - x0, y1 - y0)
    return (x1 - x0) / L, (y1 - y0) / L


def physical_members(model: Model) -> tuple[tuple[int, ...], ...]:
    """Barras físicas: cadenas de barras colineales consecutivas, para pandeo y PLT.

    Dos barras se unen en un nudo si sólo ellas concurren a él, el nudo no tiene apoyo ni resorte (puntos
    de arriostramiento), son colineales y siguen en la misma dirección, y tienen igual sección y material.
    Las rótulas internas no cortan la cadena. Cada cadena es una tupla creciente de índices de barra;
    las cadenas se ordenan por su primera barra.
    """
    parent = list(range(len(model.members)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    at_node: dict[int, list[int]] = {}
    for k, mem in enumerate(model.members):
        at_node.setdefault(mem.i, []).append(k)
        at_node.setdefault(mem.j, []).append(k)
    braced = {s.node for s in model.supports} | {sp.node for sp in model.springs}
    for node, ms in at_node.items():
        if len(ms) != 2 or node in braced:
            continue
        a, b = ms
        ma, mb = model.members[a], model.members[b]
        if ma.section != mb.section or ma.material != mb.material:
            continue
        ua, ub = _unit_away(model, a, node), _unit_away(model, b, node)
        if ua[0] * ub[0] + ua[1] * ub[1] > -1.0 + 1e-9:  # no colineales u opuestas
            continue
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for k in range(len(model.members)):
        groups.setdefault(find(k), []).append(k)
    return tuple(sorted(tuple(g) for g in groups.values()))


@dataclass(frozen=True, slots=True)
class MemberStabilityCheck:
    """Estabilidad de una barra física: pandeo flexional + PLT + interacción H1 (AISC 360-16 §H1.1).

    `column.phi_c` y `ltb.phi_b` son el factor de resistencia usado (1/FS_min en `check_safety`; φ en ELU).
    """

    members: tuple[int, ...]  # barras del modelo que forman la barra física
    length: float  # longitud de la barra física [mm]
    N_comp: float  # compresión máxima [N] (≥ 0; 0 si la barra sólo está traccionada)
    M_max: float  # |M| máximo [N·mm]
    column: ColumnBucklingCheck  # con K·L de la barra física
    ltb: BeamLTBCheck  # con Lb = mín(unbraced_length, L)

    @property
    def utilization(self) -> float:
        """Interacción H1 con resistencias disponibles (Pn, Mn por φ o 1/FS); > 1 no verifica."""
        return interaction_h1(self.N_comp, self.column.phi_Pn, self.M_max, self.ltb.phi_Mn)

    @property
    def utilization_nominal(self) -> float:
        """Interacción H1 con las resistencias nominales Pn y Mn; > 1 = inestable con estas solicitaciones."""
        return interaction_h1(self.N_comp, self.column.Pn, self.M_max, self.ltb.Mn)

    @property
    def mode(self) -> str:
        """Modo de inestabilidad, en castellano (se ignoran solicitaciones < 1e-9 de la resistencia)."""
        has_n = self.N_comp > 1e-9 * self.column.Pn
        has_m = self.M_max > 1e-9 * self.ltb.Mn
        if has_n and has_m:
            return "flexocompresión (pandeo flexional + PLT)"
        return "pandeo flexional" if has_n else "pandeo lateral-torsional"


def _members_text(members: tuple[int, ...]) -> str:
    if len(members) == 1:
        return f"barra {members[0]}"
    contiguous = members == tuple(range(members[0], members[-1] + 1))
    return (f"barras {members[0]}–{members[-1]}" if contiguous
            else "barras " + ", ".join(str(m) for m in members))


def _stability_issue(cs: MemberStabilityCheck, sigma_utilization: float, factor: str) -> str:
    """Texto de `issues` para una barra física que no verifica estabilidad. `factor`: "/FS" o "φ·"."""
    head = "Inestabilidad" if cs.utilization_nominal > 1.0 else "Estabilidad"
    cap_p = "φ·Pn" if factor == "φ·" else "Pn/FS"
    cap_m = "φ·Mn" if factor == "φ·" else "Mn/FS"
    text = (
        f"{head}: {cs.mode} en {_members_text(cs.members)} (L = {cs.length * _MM_TO_M_TEXT:.2f} m), "
        f"utilización {cs.utilization:.0%} (AISC 360 H1): N = {cs.N_comp * _N_TO_KN_TEXT:.1f} kN frente a "
        f"{cap_p} = {cs.column.phi_Pn * _N_TO_KN_TEXT:.1f} kN y M = {cs.M_max * _NMM_TO_KNM_TEXT:.1f} kN·m "
        f"frente a {cap_m} = {cs.ltb.phi_Mn * _NMM_TO_KNM_TEXT:.1f} kN·m; KL/r = {cs.column.slenderness:.0f} "
        f"({cs.column.regime.value}), Lb = {cs.ltb.Lb * _MM_TO_M_TEXT:.2f} m ({cs.ltb.zone.value})."
    )
    if cs.utilization > sigma_utilization:
        text += (f" Gobierna la inestabilidad frente a la fluencia de la sección "
                 f"(utilización por tensiones {sigma_utilization:.0%}).")
    return text


def _stability_notes(checks: Sequence[MemberStabilityCheck]) -> tuple[str, ...]:
    out: list[str] = []
    for cs in checks:
        where = _members_text(cs.members)
        if cs.ltb.zone is LTBZone.NOT_EVALUATED:
            out.append(f"PLT no evaluado en {where} ({cs.ltb.section}): {cs.ltb.method}.")
        if cs.N_comp > 1e-9 * cs.column.Pn and not cs.column.slenderness_ok:
            out.append(f"Esbeltez KL/r = {cs.column.slenderness:.0f} > {SLENDERNESS_MAX:g} en {where} "
                       f"(límite recomendado AISC 360-16 §E2).")
    return tuple(out)


def _stability_checks(
    model: Model, results: Results, phi: float, unbraced_length: float | None, K: float,
) -> tuple[MemberStabilityCheck, ...]:
    """Una `MemberStabilityCheck` por barra física, con las solicitaciones de `results`."""
    by_member = {mr.member: mr for mr in results.members}
    out: list[MemberStabilityCheck] = []
    for chain in physical_members(model):
        mem = model.members[chain[0]]
        L = sum(model.member_length(k) for k in chain)
        Lb = L if unbraced_length is None else min(unbraced_length, L)
        N_comp = max(0.0, max(-float(np.min(by_member[k].N)) for k in chain))
        M_max = max(float(np.max(np.abs(by_member[k].M))) for k in chain)
        Fy, E = mem.material.Sy, mem.material.E
        out.append(MemberStabilityCheck(
            members=chain, length=L, N_comp=N_comp, M_max=M_max,
            column=column_buckling_capacity(mem.section, L, K, Fy, phi, E=E, N_comp=N_comp),
            ltb=beam_ltb_capacity(mem.section, Lb, Fy, CB_DEFAULT, phi, E=E, M_u=M_max),
        ))
    return tuple(out)


class Status(str, Enum):
    OK = "OK"  # resistencia y servicio verifican
    ALERT = "ALERTA"  # 1 <= FS < FS_min, o flecha mayor a la admisible (sin fluencia)
    FAIL = "FLUENCIA"  # FS < 1 (domina sobre la flecha)


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    fs: float  # factor de seguridad mínimo de la estructura
    fs_min: float  # FS admisible requerido
    sigma_max: float  # |σ| normal máxima de la barra crítica (fibra extrema) [MPa]
    sy: float  # fluencia del material crítico [MPa]
    member: int  # barra crítica
    x: float  # posición local en la barra crítica [mm]
    point: tuple[float, float]  # posición global [mm]
    status: Status  # veredicto global (resistencia + servicio)
    utilization: float  # σ_vM / (Sy / FS_min), >1 no verifica (σ_vM = σ_max en flexión dominante)
    # ---- Servicio (ELS) — campos agregados al final (AGENTS.md §5) ---- #
    delta_max: float = 0.0  # |δ| máxima de la estructura [mm]
    deflection_ratio: float = math.inf  # N = L / δ_max (flecha relativa L/N)
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT  # N admisible; 0 = sin límite
    deflection_ok: bool = True  # N >= N admisible  (δ_max <= L / N_adm)
    reference_length: float = 0.0  # L usada para L/δ [mm]
    # ---- Corte y von Mises — agregados al final (AGENTS.md §5) ---- #
    tau_max: float = 0.0  # τ máxima de la barra crítica = |V|·Q/(I·t) [MPa]
    sigma_vm_max: float = 0.0  # σ_vM máxima de la barra crítica [MPa]; FS = Sy / σ_vM
    # ---- Estabilidad (pandeo / PLT) — agregado al final (AGENTS.md §5) ---- #
    stability: tuple[MemberStabilityCheck, ...] = ()  # una por barra física; () = no evaluada

    @property
    def critical_stability(self) -> MemberStabilityCheck | None:
        """Barra física de máxima utilización de estabilidad (None si no se evaluó)."""
        return max(self.stability, key=lambda c: c.utilization, default=None)

    @property
    def stability_utilization(self) -> float:
        """Utilización H1 máxima (resistencias / FS_min o ·φ); 0 si no se evaluó; > 1 no verifica."""
        cs = self.critical_stability
        return cs.utilization if cs is not None else 0.0

    @property
    def stability_ok(self) -> bool:
        """Verifica estabilidad (o no se evaluó)."""
        return self.stability_utilization <= 1.0

    @property
    def buckling_failure(self) -> bool:
        """Alguna barra física es inestable con las resistencias nominales (utilización nominal > 1)."""
        return any(c.utilization_nominal > 1.0 for c in self.stability)

    @property
    def stability_governs(self) -> bool:
        """La inestabilidad (pandeo / PLT) supera a la utilización por tensiones de la sección."""
        return bool(self.stability) and self.stability_utilization > self.utilization

    @property
    def uls_ok(self) -> bool:
        """Verifican resistencia de la sección y estabilidad."""
        return self.strength_ok and self.stability_ok

    @property
    def stability_notes(self) -> tuple[str, ...]:
        """Avisos de estabilidad que no cambian el veredicto (PLT no evaluado, KL/r > 200)."""
        return _stability_notes(self.stability)

    @property
    def ok(self) -> bool:
        """Verifica resistencia y servicio."""
        return self.status is Status.OK

    @property
    def shear_governs(self) -> bool:
        """El von Mises del eje neutro (corte) supera a la tensión normal de la fibra extrema."""
        return self.sigma_vm_max > self.sigma_max * (1.0 + _SHEAR_GOVERNS_RTOL)

    def _stress_text(self, sub: str = "") -> str:
        """'σ máx = … MPa' (flexión) o 'σvM = … MPa (corte, τ máx = … MPa)' si gobierna el corte."""
        if self.shear_governs:
            vm = f"σvM,{sub}" if sub else "σvM"
            return (f"{vm} = {self.sigma_vm_max:.1f} MPa (corte en el eje neutro, "
                    f"τ{sub} máx = {self.tau_max:.1f} MPa)")
        return f"σ{sub} máx = {self.sigma_max:.1f} MPa" if not sub else f"σ{sub} = {self.sigma_max:.1f} MPa"

    @property
    def strength_ok(self) -> bool:
        """FS >= FS_min (sólo resistencia)."""
        return self.fs >= self.fs_min

    @property
    def delta_adm(self) -> float:
        """Flecha admisible δ_adm = L / N_adm [mm]; infinito si no hay límite."""
        if self.deflection_limit_ratio <= 0.0:
            return math.inf
        return self.reference_length / self.deflection_limit_ratio

    @property
    def issues(self) -> tuple[str, ...]:
        """Motivos por los que no es un OK limpio, en castellano (la UI los muestra tal cual)."""
        out: list[str] = []
        where = _where(self.member, self.point)
        if self.fs < 1.0:
            out.append(f"Fluencia: {self._stress_text()} supera Sy = {self.sy:g} MPa "
                       f"en {where}.")
        elif not self.strength_ok:
            out.append(f"Resistencia: FS = {self.fs:.2f} < FS mín. = {self.fs_min:g} "
                       f"({self._stress_text()} en {where}).")
        cs = self.critical_stability
        if cs is not None and not self.stability_ok:
            out.append(_stability_issue(cs, self.utilization, "/FS"))
        if not self.deflection_ok:
            out.append(f"Servicio: flecha δ máx = {self.delta_max:.1f} mm supera δ adm = "
                       f"L/{self.deflection_limit_ratio:g} = {self.delta_adm:.1f} mm "
                       f"(L/δ = L/{self.deflection_ratio:,.0f}).")
        return tuple(out)


def default_reference_length(model: Model) -> float:
    """Longitud de referencia para L/δ [mm]: ancho horizontal del modelo (luz).

    Si el modelo no tiene extensión horizontal (p. ej. una columna aislada), usa la
    barra más larga.
    """
    xs = [x for x, _ in model.nodes]
    return max(xs) - min(xs) or max(model.member_length(m) for m in range(len(model.members)))


def check_safety(
    model: Model,
    results: Results,
    fs_min: float = FS_MIN_DEFAULT,
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT,
    reference_length: float | None = None,
    *,
    stability: bool = False,
    unbraced_length: float | None = None,
    effective_length_factor_k: float = 1.0,
) -> SafetyCheck:
    """Verificación de resistencia (FS = Sy / σ_vM), servicio (L/δ_max ≥ N) y, opcional, estabilidad.

    σ_vM = máx(σ_max, √((N/A)² + 3τ²)) en cada estación (`member_stresses`); con flexión dominante
    σ_vM = σ_max y FS = Sy / σ_max como siempre. La barra crítica es la de menor FS; `sigma_max`,
    `tau_max` y `sigma_vm_max` son los máximos de esa barra y la ubicación es la de σ_vM máxima.

    Args:
        fs_min: factor de seguridad mínimo admisible.
        deflection_limit_ratio: N de la flecha admisible L/N (300 → L/300). 0 = sin límite de flecha.
        reference_length: L para L/δ [mm]; por defecto `default_reference_length(model)`.
        stability: evalúa pandeo flexional y PLT por barra física (ver docstring del módulo) con
            Pc = Pn/FS_min y Mc = Mn/FS_min. Por defecto False: resultado idéntico al de siempre.
        unbraced_length: separación de arriostramientos laterales Lb [mm] para el PLT; se acota a la
            longitud de cada barra física (por defecto, esa longitud). 0 = arriostramiento continuo.
        effective_length_factor_k: K del pandeo flexional (KL con la longitud de la barra física).

    Veredicto: FLUENCIA si FS < 1 o si alguna barra es inestable con resistencias nominales (el texto de
    `issues` dice cuál); ALERTA si FS < fs_min, si la estabilidad no verifica o si δ_max > L/N; OK si todo
    verifica.

    Raises:
        ValueError: fs_min ≤ 0 (o NaN), N < 0 o L ≤ 0; con `stability`: fs_min < 1, K ≤ 0 o Lb < 0.
    """
    if not fs_min > 0.0:  # también rechaza NaN
        raise ValueError("El factor de seguridad admisible debe ser estrictamente positivo.")
    if deflection_limit_ratio < 0.0:
        raise ValueError("El límite de flecha L/N debe ser N ≥ 0 (0 = sin límite).")
    L_ref = default_reference_length(model) if reference_length is None else reference_length
    if L_ref <= 0.0:
        raise ValueError("La longitud de referencia para la flecha debe ser positiva [mm].")
    if stability:
        if fs_min < 1.0:
            raise ValueError("Para verificar estabilidad el factor de seguridad admisible debe ser ≥ 1.")
        if not (effective_length_factor_k > 0.0 and math.isfinite(effective_length_factor_k)):
            raise ValueError("El factor de longitud efectiva K debe ser positivo.")
        if unbraced_length is not None and not (unbraced_length >= 0.0 and math.isfinite(unbraced_length)):
            raise ValueError("La longitud no arriostrada Lb debe ser ≥ 0 [mm].")

    # (fs, tensiones, Sy, barra, estación de σ_vM máx)
    best: tuple[float, MemberStresses, float, MemberResult, int] | None = None
    for mr in results.members:
        member = model.members[mr.member]
        sy = member.material.Sy
        st = member_stresses(member.section, mr)
        k = int(np.argmax(st.sigma_vm))  # flexión dominante: misma estación que argmax(σ_max)
        svm = float(st.sigma_vm[k])
        fs = sy / svm if svm > 1e-12 else math.inf
        if best is None or fs < best[0]:
            best = (fs, st, sy, mr, k)
    if best is None:
        raise ValueError("Resultados sin barras.")
    fs, st, sy, mr, k = best
    smax = float(np.max(st.sigma_max))
    svm = float(st.sigma_vm[k])

    delta = abs(results.extreme("deflection").value)
    ratio = L_ref / delta if delta > 1e-12 else math.inf
    deflection_ok = deflection_limit_ratio == 0.0 or ratio >= deflection_limit_ratio

    stab = (_stability_checks(model, results, 1.0 / fs_min, unbraced_length, effective_length_factor_k)
            if stability else ())
    stab_u = max((c.utilization for c in stab), default=0.0)
    if fs < 1.0 or any(c.utilization_nominal > 1.0 for c in stab):
        status = Status.FAIL
    elif fs < fs_min or stab_u > 1.0 or not deflection_ok:
        status = Status.ALERT
    else:
        status = Status.OK

    return SafetyCheck(
        fs=fs, fs_min=fs_min, sigma_max=smax, sy=sy, member=mr.member,
        x=float(mr.x[k]), point=(float(mr.points[k, 0]), float(mr.points[k, 1])),
        status=status, utilization=svm * fs_min / sy,
        delta_max=delta, deflection_ratio=ratio, deflection_limit_ratio=deflection_limit_ratio,
        deflection_ok=deflection_ok, reference_length=L_ref,
        tau_max=float(np.max(st.tau)), sigma_vm_max=svm, stability=stab,
    )


@dataclass(frozen=True, slots=True)
class CombinationCheck:
    """Verificación por combinaciones de carga: ELU por envolvente + ELS de servicio.

    Cada `SafetyCheck` de ELU usa FS_min = 1/φ, de modo que FS ≥ FS_min ⇔ σ_u ≤ φ·Sy y
    `utilization` = σ_u / (φ·Sy). El de ELS sólo se usa por su flecha.
    """

    phi: float  # factor de resistencia φ
    strength_checks: tuple[tuple[LoadCombination, SafetyCheck], ...]  # una por combinación ELU
    governing: LoadCombination  # combinación ELU de máxima utilización (gobierna resistencia)
    service_combination: LoadCombination  # combinación de servicio (cargas sin mayorar)
    service_check: SafetyCheck  # flecha bajo `service_combination`
    governing_results: Results  # solicitaciones mayoradas de la combinación gobernante
    service_results: Results  # resultados bajo la combinación de servicio
    # ---- agregado al final (AGENTS.md §5) ---- #
    uls_results: tuple[Results, ...] = ()  # resultados de cada combinación ELU (orden de strength_checks)

    @property
    def strength(self) -> SafetyCheck:
        """Verificación de resistencia de la combinación gobernante."""
        return next(chk for comb, chk in self.strength_checks if comb is self.governing)

    @property
    def utilization(self) -> float:
        """σ_u / (φ·Sy) de la combinación gobernante; > 1 no verifica."""
        return self.strength.utilization

    @property
    def strength_ok(self) -> bool:
        return self.strength.strength_ok

    @property
    def tau_max(self) -> float:
        """τ máxima [MPa] en la barra crítica de la combinación gobernante (cargas mayoradas)."""
        return self.strength.tau_max

    @property
    def sigma_vm_max(self) -> float:
        """σ_vM máxima [MPa] de la combinación gobernante; ELU verifica si σ_vM ≤ φ·Sy."""
        return self.strength.sigma_vm_max

    @property
    def shear_governs(self) -> bool:
        """El corte combinado gobierna la resistencia de la combinación ELU."""
        return self.strength.shear_governs

    @property
    def stability_governing(self) -> tuple[LoadCombination, SafetyCheck]:
        """(combinación, verificación) ELU de máxima utilización de estabilidad."""
        return max(self.strength_checks, key=lambda cc: cc[1].stability_utilization)

    @property
    def stability_utilization(self) -> float:
        """Utilización H1 máxima entre las combinaciones ELU (0 si no se evaluó); > 1 no verifica."""
        return self.stability_governing[1].stability_utilization

    @property
    def stability_ok(self) -> bool:
        return self.stability_utilization <= 1.0

    @property
    def stability_governs(self) -> bool:
        """En la combinación gobernante, la inestabilidad supera a la utilización por tensiones."""
        return self.strength.stability_governs

    @property
    def delta_max(self) -> float:
        """|δ| máxima bajo la combinación de servicio [mm]."""
        return self.service_check.delta_max

    @property
    def deflection_ok(self) -> bool:
        return self.service_check.deflection_ok

    @property
    def status(self) -> Status:
        """FLUENCIA si σ_u > Sy; ALERTA si σ_u > φ·Sy, la estabilidad no verifica o la flecha de servicio
        excede L/N; si no, OK."""
        if self.strength.fs < 1.0:
            return Status.FAIL
        if not self.strength_ok or not self.stability_ok or not self.deflection_ok:
            return Status.ALERT
        return Status.OK

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @property
    def verdict(self) -> tuple[Status, str]:
        """Veredicto global para mostrar (UI y PDF): (severidad, rótulo).

        - (FAIL, "FLUENCIA"): σ_u > Sy en la combinación gobernante.
        - (FAIL, "NO VERIFICA"): φ·Sy < σ_u ≤ Sy, o la estabilidad no verifica (Pu, Mu frente a φPn, φMn,
          AISC H1); en LRFD no verificar ELU es una falla, no una alerta.
        - (ALERT, "ALERTA"): ELU verifica pero la flecha de servicio supera L/N.
        - (OK, "VERIFICA"): verifican ELU y ELS.
        La severidad sólo elige el color; el detalle se lee de `issues`.
        """
        if self.status is Status.FAIL:
            return Status.FAIL, "FLUENCIA"
        if not self.strength_ok or not self.stability_ok:
            return Status.FAIL, "NO VERIFICA"
        if not self.deflection_ok:
            return Status.ALERT, "ALERTA"
        return Status.OK, "VERIFICA"

    @property
    def issues(self) -> tuple[str, ...]:
        """Motivos de un veredicto no-OK, en castellano e indicando la combinación."""
        out: list[str] = []
        st, name = self.strength, self.governing.name
        where = _where(st.member, st.point)
        if st.fs < 1.0:
            out.append(f"ELU {name}: {st._stress_text('u')} supera Sy = {st.sy:g} MPa "
                       f"(fluencia) en {where}.")
        elif not st.strength_ok:
            out.append(f"ELU {name}: {st._stress_text('u')} > φ·Sy = {self.phi:g}·{st.sy:g} = "
                       f"{self.phi * st.sy:.1f} MPa (utilización {st.utilization:.0%}) en {where}.")
        comb_s, chk_s = self.stability_governing
        cs = chk_s.critical_stability
        if cs is not None and not chk_s.stability_ok:
            out.append(f"ELU {comb_s.name}: " + _stability_issue(cs, chk_s.utilization, "φ·"))
        sv = self.service_check
        if not sv.deflection_ok:
            out.append(f"ELS {self.service_combination.name}: flecha δ máx = {sv.delta_max:.1f} mm "
                       f"supera δ adm = L/{sv.deflection_limit_ratio:g} = {sv.delta_adm:.1f} mm "
                       f"(L/δ = L/{sv.deflection_ratio:,.0f}).")
        return tuple(out)


def check_combinations(
    model: Model,
    phi: float = PHI_DEFAULT,
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT,
    reference_length: float | None = None,
    uls_combinations: Sequence[LoadCombination] = ULS_COMBINATIONS,
    service_combination: LoadCombination = COMB_SERVICE_D_L,
    n_stations: int = 41,
    case_results: dict[LoadCase, Results] | None = None,
    *,
    stability: bool = False,
    unbraced_length: float | None = None,
    effective_length_factor_k: float = 1.0,
) -> CombinationCheck:
    """Resuelve D y L una vez y verifica ELU (envolvente) y ELS por combinaciones.

    `case_results`: respuestas por hipótesis ya resueltas (`solve_load_cases(model)`); si se dan,
    no se vuelve a ensamblar ni resolver K (la UI resuelve una sola vez por cambio de modelo).

    Args:
        phi: factor de resistencia φ (0 < φ ≤ 1); ELU verifica si σ_u ≤ φ·Sy.
        deflection_limit_ratio, reference_length: como en `check_safety` (flecha L/N de servicio).
        uls_combinations: combinaciones de resistencia (por defecto 1.4D y 1.2D + 1.6L).
        service_combination: combinación de servicio (por defecto 1.0D + 1.0L).
        stability, unbraced_length, effective_length_factor_k: como en `check_safety`, sólo en ELU
            (Pu, Mu frente a φc·Pn y φb·Mn, interacción H1). Por defecto sin estabilidad.

    Raises:
        ValueError: φ fuera de (0, 1], combinaciones vacías o de estado límite equivocado, y los
            de `check_safety`. StructuralError: estructura inestable.
    """
    if not 0.0 < phi <= 1.0:  # también rechaza NaN
        raise ValueError("El factor de resistencia φ debe estar en el intervalo (0; 1].")
    if not uls_combinations:
        raise ValueError("Se necesita al menos una combinación de estado límite último (ELU).")
    for comb in uls_combinations:
        if comb.limit_state != ULS:
            raise ValueError(f"La combinación {comb.name!r} no es de resistencia ({ULS}).")
    if service_combination.limit_state != SLS:
        raise ValueError(f"La combinación {service_combination.name!r} no es de servicio ({SLS}).")

    combos = [*uls_combinations, service_combination]
    results = (solve_combinations(model, combos, n_stations) if case_results is None
               else combine_cases(case_results, combos))
    strength_checks = tuple(
        (comb, check_safety(model, res, fs_min=1.0 / phi, deflection_limit_ratio=0.0,
                            reference_length=reference_length, stability=stability,
                            unbraced_length=unbraced_length,
                            effective_length_factor_k=effective_length_factor_k))
        for comb, res in zip(uls_combinations, results, strict=False)
    )
    k_gov = max(range(len(strength_checks)),
                key=lambda k: max(strength_checks[k][1].utilization,
                                  strength_checks[k][1].stability_utilization))
    service_check = check_safety(model, results[-1], fs_min=1.0 / phi,
                                 deflection_limit_ratio=deflection_limit_ratio,
                                 reference_length=reference_length)
    return CombinationCheck(
        phi=phi, strength_checks=strength_checks, governing=strength_checks[k_gov][0],
        service_combination=service_combination, service_check=service_check,
        governing_results=results[k_gov], service_results=results[-1],
        uls_results=tuple(results[:-1]),
    )


def required_section_modulus(M_max: float, sy: float, fs: float = FS_MIN_DEFAULT) -> float:
    """W requerido [mm³] para |M_max| [N·mm] con FS dado (predimensionamiento a flexión pura)."""
    return abs(M_max) * fs / sy
