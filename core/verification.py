"""Verificación estructural (postprocesamiento). Unidades: MPa, mm.

Criterios MVP:
- Resistencia: tensión normal elástica máxima (N/A ± M·c/I) frente a la fluencia
  del material de cada barra. FS = Sy / σ_max ≥ FS_min.
- Servicio (ELS): flecha relativa L/δ_max ≥ N (por defecto L/300), con δ_max el
  desplazamiento total máximo de la estructura y L la longitud de referencia
  (por defecto, el ancho horizontal del modelo = luz de la viga o del pórtico).

Combinaciones de carga (`check_combinations`, hipótesis D y L):
- ELU (resistencia), envolvente de 1.4D y 1.2D + 1.6L (LRFD): σ_u ≤ φ·Sy con φ = 0.90, sobre la
  tensión normal elástica de las solicitaciones mayoradas (criterio de primera fluencia,
  conservador frente al momento plástico). Gobierna la combinación de máxima utilización
  σ_u / (φ·Sy), que es el máximo de la envolvente.
- ELS (servicio): flecha bajo 1.0D + 1.0L (cargas sin mayorar) frente a L/N.

No incluye pandeo global, pandeo lateral-torsional, abolladura local,
corte combinado (von Mises) ni fatiga.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.model import COMB_SERVICE_D_L, SLS, ULS, ULS_COMBINATIONS, LoadCase, LoadCombination, Model
from core.solver import MemberResult, Results, combine_cases, solve_combinations

FS_MIN_DEFAULT = 1.5

# mm -> m SÓLO para redactar las coordenadas de `issues` (texto listo para mostrar). Excepción de
# presentación documentada en AGENTS.md §4: ningún número convertido sale de este módulo.
_MM_TO_M_TEXT = 1e-3


def _where(member: int, point: tuple[float, float]) -> str:
    """Ubicación para los mensajes de `issues`: 'barra k, (x; y) m'."""
    return f"barra {member}, ({point[0] * _MM_TO_M_TEXT:.2f}; {point[1] * _MM_TO_M_TEXT:.2f}) m"

# Flecha admisible por defecto L/300: valor usual de predimensionamiento para vigas.
# Los límites de servicio dependen del uso y de la norma (AISC 360 cap. L y CIRSOC 301
# remiten a criterios de servicio por tipo de elemento): ajustar con `deflection_limit_ratio`.
DEFLECTION_LIMIT_DEFAULT = 300.0

# Factor de resistencia LRFD a flexión φb = 0.90: AISC 360-16 §F1 y CIRSOC 301-2005 §F.1.
PHI_DEFAULT = 0.90


class Status(str, Enum):
    OK = "OK"  # resistencia y servicio verifican
    ALERT = "ALERTA"  # 1 <= FS < FS_min, o flecha mayor a la admisible (sin fluencia)
    FAIL = "FLUENCIA"  # FS < 1 (domina sobre la flecha)


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    fs: float  # factor de seguridad mínimo de la estructura
    fs_min: float  # FS admisible requerido
    sigma_max: float  # |σ| máxima [MPa]
    sy: float  # fluencia del material crítico [MPa]
    member: int  # barra crítica
    x: float  # posición local en la barra crítica [mm]
    point: tuple[float, float]  # posición global [mm]
    status: Status  # veredicto global (resistencia + servicio)
    utilization: float  # σ_max / (Sy / FS_min), >1 no verifica
    # ---- Servicio (ELS) — campos agregados al final (AGENTS.md §5) ---- #
    delta_max: float = 0.0  # |δ| máxima de la estructura [mm]
    deflection_ratio: float = math.inf  # N = L / δ_max (flecha relativa L/N)
    deflection_limit_ratio: float = DEFLECTION_LIMIT_DEFAULT  # N admisible; 0 = sin límite
    deflection_ok: bool = True  # N >= N admisible  (δ_max <= L / N_adm)
    reference_length: float = 0.0  # L usada para L/δ [mm]

    @property
    def ok(self) -> bool:
        """Verifica resistencia y servicio."""
        return self.status is Status.OK

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
            out.append(f"Fluencia: σ máx = {self.sigma_max:.1f} MPa supera Sy = {self.sy:g} MPa "
                       f"en {where}.")
        elif not self.strength_ok:
            out.append(f"Resistencia: FS = {self.fs:.2f} < FS mín. = {self.fs_min:g} "
                       f"(σ máx = {self.sigma_max:.1f} MPa en {where}).")
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
) -> SafetyCheck:
    """Verificación de resistencia (FS = Sy / σ_max) y de servicio (L/δ_max ≥ N).

    Args:
        fs_min: factor de seguridad mínimo admisible.
        deflection_limit_ratio: N de la flecha admisible L/N (300 → L/300). 0 = sin límite de flecha.
        reference_length: L para L/δ [mm]; por defecto `default_reference_length(model)`.

    Veredicto: FLUENCIA si FS < 1; ALERTA si FS < fs_min o δ_max > L/N; OK si ambos verifican.

    Raises:
        ValueError: fs_min ≤ 0 (o NaN), N < 0 o L ≤ 0.
    """
    if not fs_min > 0.0:  # también rechaza NaN
        raise ValueError("El factor de seguridad admisible debe ser estrictamente positivo.")
    if deflection_limit_ratio < 0.0:
        raise ValueError("El límite de flecha L/N debe ser N ≥ 0 (0 = sin límite).")
    L_ref = default_reference_length(model) if reference_length is None else reference_length
    if L_ref <= 0.0:
        raise ValueError("La longitud de referencia para la flecha debe ser positiva [mm].")

    best: tuple[float, float, float, MemberResult, int] | None = None  # (fs, σ, Sy, barra, estación)
    for mr in results.members:
        sy = model.members[mr.member].material.Sy
        sig = mr.sigma_abs
        k = int(np.argmax(sig))
        smax = float(sig[k])
        fs = sy / smax if smax > 1e-12 else math.inf
        if best is None or fs < best[0]:
            best = (fs, smax, sy, mr, k)
    if best is None:
        raise ValueError("Resultados sin barras.")
    fs, smax, sy, mr, k = best

    delta = abs(results.extreme("deflection").value)
    ratio = L_ref / delta if delta > 1e-12 else math.inf
    deflection_ok = deflection_limit_ratio == 0.0 or ratio >= deflection_limit_ratio

    if fs < 1.0:
        status = Status.FAIL
    elif fs < fs_min or not deflection_ok:
        status = Status.ALERT
    else:
        status = Status.OK

    return SafetyCheck(
        fs=fs, fs_min=fs_min, sigma_max=smax, sy=sy, member=mr.member,
        x=float(mr.x[k]), point=(float(mr.points[k, 0]), float(mr.points[k, 1])),
        status=status, utilization=smax * fs_min / sy,
        delta_max=delta, deflection_ratio=ratio, deflection_limit_ratio=deflection_limit_ratio,
        deflection_ok=deflection_ok, reference_length=L_ref,
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
    def delta_max(self) -> float:
        """|δ| máxima bajo la combinación de servicio [mm]."""
        return self.service_check.delta_max

    @property
    def deflection_ok(self) -> bool:
        return self.service_check.deflection_ok

    @property
    def status(self) -> Status:
        """FLUENCIA si σ_u > Sy; ALERTA si σ_u > φ·Sy o la flecha de servicio excede L/N; si no, OK."""
        if self.strength.fs < 1.0:
            return Status.FAIL
        if not self.strength_ok or not self.deflection_ok:
            return Status.ALERT
        return Status.OK

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @property
    def verdict(self) -> tuple[Status, str]:
        """Veredicto global para mostrar (UI y PDF): (severidad, rótulo).

        - (FAIL, "FLUENCIA"): σ_u > Sy en la combinación gobernante.
        - (FAIL, "NO VERIFICA"): φ·Sy < σ_u ≤ Sy (en LRFD no verificar ELU es una falla, no una alerta).
        - (ALERT, "ALERTA"): ELU verifica pero la flecha de servicio supera L/N.
        - (OK, "VERIFICA"): verifican ELU y ELS.
        La severidad sólo elige el color; el detalle se lee de `issues`.
        """
        if self.status is Status.FAIL:
            return Status.FAIL, "FLUENCIA"
        if not self.strength_ok:
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
            out.append(f"ELU {name}: σu = {st.sigma_max:.1f} MPa supera Sy = {st.sy:g} MPa "
                       f"(fluencia) en {where}.")
        elif not st.strength_ok:
            out.append(f"ELU {name}: σu = {st.sigma_max:.1f} MPa > φ·Sy = {self.phi:g}·{st.sy:g} = "
                       f"{self.phi * st.sy:.1f} MPa (utilización {st.utilization:.0%}) en {where}.")
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
) -> CombinationCheck:
    """Resuelve D y L una vez y verifica ELU (envolvente) y ELS por combinaciones.

    `case_results`: respuestas por hipótesis ya resueltas (`solve_load_cases(model)`); si se dan,
    no se vuelve a ensamblar ni resolver K (la UI resuelve una sola vez por cambio de modelo).

    Args:
        phi: factor de resistencia φ (0 < φ ≤ 1); ELU verifica si σ_u ≤ φ·Sy.
        deflection_limit_ratio, reference_length: como en `check_safety` (flecha L/N de servicio).
        uls_combinations: combinaciones de resistencia (por defecto 1.4D y 1.2D + 1.6L).
        service_combination: combinación de servicio (por defecto 1.0D + 1.0L).

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
                            reference_length=reference_length))
        for comb, res in zip(uls_combinations, results, strict=False)
    )
    k_gov = max(range(len(strength_checks)), key=lambda k: strength_checks[k][1].utilization)
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
