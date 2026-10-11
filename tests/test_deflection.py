"""Estado límite de servicio: flecha admisible L/N (check_safety y suggest_section).

Caso base: viga simplemente apoyada IPE 240, L = 8 m, carga uniforme, sin peso propio.
Referencias analíticas (Gere, Mechanics of Materials, tabla de vigas):
    M_máx = q·L²/8 ;  σ_máx = M_máx / W ;  δ_máx = 5·q·L⁴ / (384·E·I)  (centro del tramo).
"""

from __future__ import annotations

import math

import pytest

from core.builders import BeamDistLoad, BeamSupport, build_beam
from core.design import suggest_section
from core.materials import MATERIALS, Section, get_section
from core.model import SupportType as ST
from core.solver import solve
from core.verification import DEFLECTION_LIMIT_DEFAULT, Status, check_safety

L = 8000.0  # mm
MAT = MATERIALS["ASTM A36"]  # Sy = 250 MPa, E = 200 000 MPa


def _beam(sec: Section, q: float):
    return build_beam(L, [BeamSupport(0.0, ST.PINNED), BeamSupport(L, ST.ROLLER)], sec, MAT,
                      dist_loads=[BeamDistLoad(0.0, L, q, q)])


def _delta(q: float, I: float) -> float:
    """δ_máx = 5·|q|·L⁴ / (384·E·I) [mm]."""
    return 5.0 * abs(q) * L**4 / (384.0 * MAT.E * I)


def test_strength_ok_but_deflection_fails():
    """IPE 240, q = 5 N/mm: σ = qL²/8/W = 123 MPa < Sy/1,5 = 167 MPa, pero δ = 34,3 mm > L/300 = 26,7 mm."""
    sec, q = get_section("IPE 240"), -5.0
    m = _beam(sec, q)
    chk = check_safety(m, solve(m), fs_min=1.5)

    sigma = abs(q) * L**2 / 8.0 / sec.W
    assert chk.sigma_max == pytest.approx(sigma, rel=1e-6)
    assert sigma < MAT.Sy / 1.5
    assert chk.strength_ok

    delta = _delta(q, sec.I)
    assert delta > L / 300.0
    assert chk.delta_max == pytest.approx(delta, rel=1e-6)
    assert chk.deflection_ratio == pytest.approx(L / delta, rel=1e-6)
    assert chk.deflection_limit_ratio == DEFLECTION_LIMIT_DEFAULT == 300.0
    assert chk.delta_adm == pytest.approx(L / 300.0)
    assert chk.reference_length == pytest.approx(L)
    assert chk.deflection_ok is False

    # Veredicto global: no fluye, pero no es un OK limpio
    assert chk.status is Status.ALERT
    assert not chk.ok
    assert len(chk.issues) == 1 and chk.issues[0].startswith("Servicio")


def test_strength_and_deflection_both_ok():
    """IPE 240, q = 3 N/mm: σ = 74 MPa (FS = 3,4) y δ = 20,6 mm ≤ L/300 = 26,7 mm."""
    sec, q = get_section("IPE 240"), -3.0
    m = _beam(sec, q)
    chk = check_safety(m, solve(m))
    assert chk.delta_max == pytest.approx(_delta(q, sec.I), rel=1e-6)
    assert chk.delta_max <= L / 300.0
    assert chk.strength_ok and chk.deflection_ok
    assert chk.status is Status.OK and chk.ok
    assert chk.issues == ()


def test_deflection_limit_is_configurable():
    """Mismo caso que falla a L/300 (L/δ = 233): verifica a L/200 y sin límite (N = 0)."""
    m = _beam(get_section("IPE 240"), -5.0)
    r = solve(m)
    assert check_safety(m, r, deflection_limit_ratio=200.0).status is Status.OK
    no_limit = check_safety(m, r, deflection_limit_ratio=0.0)
    assert no_limit.deflection_ok and math.isinf(no_limit.delta_adm)
    assert check_safety(m, r, deflection_limit_ratio=250.0).status is Status.ALERT
    with pytest.raises(ValueError):
        check_safety(m, r, deflection_limit_ratio=-1.0)
    with pytest.raises(ValueError):
        check_safety(m, r, reference_length=0.0)


def test_reference_length_override():
    """N = L_ref / δ_máx con la longitud de referencia indicada (no la del modelo)."""
    m = _beam(get_section("IPE 240"), -5.0)
    chk = check_safety(m, solve(m), reference_length=4000.0)
    assert chk.reference_length == 4000.0
    assert chk.deflection_ratio == pytest.approx(4000.0 / chk.delta_max, rel=1e-9)


def test_yield_dominates_verdict():
    """IPE 240, q = 15 N/mm: σ = 370 MPa > Sy ⇒ FLUENCIA aunque también falle la flecha."""
    m = _beam(get_section("IPE 240"), -15.0)
    chk = check_safety(m, solve(m))
    assert chk.fs < 1.0 and not chk.deflection_ok
    assert chk.status is Status.FAIL
    assert [i.split(":")[0] for i in chk.issues] == ["Fluencia", "Servicio"]


def test_suggest_section_discards_excessive_deflection():
    """IPE, L = 8 m, q = 5 N/mm, FS ≥ 1,5.

    Resistencia: W ≥ qL²/8 · 1,5 / Sy = 240 cm³ ⇒ IPE 220 (W = 252 cm³), pero δ = 48 mm.
    Flecha L/300: I ≥ 5qL⁴·300 / (384·E·L) = 5000 cm⁴ ⇒ IPE 270 (I = 5790 cm⁴).
    Flecha absoluta 40 mm (L/200): I ≥ 3333 cm⁴ ⇒ IPE 240 (I = 3892 cm⁴).
    """
    def build(sec: Section):
        return _beam(sec, -5.0)

    only_strength = suggest_section(build, "IPE", 1.5, deflection_limit_ratio=0.0)
    assert only_strength is not None and only_strength.section.name == "IPE 220"

    default = suggest_section(build, "IPE", 1.5)
    assert default is not None and default.section.name == "IPE 270"
    assert default.check.deflection_ok and default.check.delta_max <= L / 300.0
    # el inmediato inferior no verifica la flecha
    m = build(get_section("IPE 240"))
    assert not check_safety(m, solve(m)).deflection_ok

    absolute = suggest_section(build, "IPE", 1.5, max_deflection=40.0)
    assert absolute is not None and absolute.section.name == "IPE 240"
    assert absolute.check.deflection_limit_ratio == pytest.approx(L / 40.0)
    with pytest.raises(ValueError):
        suggest_section(build, "IPE", 1.5, max_deflection=0.0)


# --------------------------------------------------------------------------- #
# suggest_section por combinaciones (mismo criterio que el veredicto de la app)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("q_D", "q_L", "lighter_fails"),
    [
        # ELU gobierna: σ_u = (1.2·q_D + 1.6·q_L)·L²/8 / W > φ·Sy en el perfil inmediato inferior
        (-10.0, -10.0, "ELU"),
        # ELS gobierna: δ(1.0D + 1.0L) = 5·(q_D + q_L)·L⁴/(384EI) > L/300 en el perfil inmediato inferior
        (-1.0, -3.0, "ELS"),
    ],
)
def test_suggest_section_with_combinations(q_D, q_L, lighter_fails):
    """Con φ: el perfil sugerido verifica `check_combinations` (ELU 1.4D / 1.2D + 1.6L con
    σ_u ≤ φ·Sy y ELS 1.0D + 1.0L ≤ L/300) y el inmediato inferior de la familia no."""
    from core.materials import SECTIONS
    from core.model import LoadCase
    from core.verification import PHI_DEFAULT, check_combinations

    def build(sec: Section):
        return build_beam(L, [BeamSupport(0.0, ST.PINNED), BeamSupport(L, ST.ROLLER)], sec, MAT,
                          dist_loads=[BeamDistLoad(0.0, L, q_D, q_D),
                                      BeamDistLoad(0.0, L, q_L, q_L, case=LoadCase.LIVE)])

    sug = suggest_section(build, "IPE", 1.5, phi=PHI_DEFAULT)
    assert sug is not None and sug.combination is not None
    assert sug.combination.ok and sug.check is sug.combination.strength
    assert check_combinations(build(sug.section)).ok
    family = sorted(SECTIONS["IPE"].values(), key=lambda s: s.A)
    lighter = family[family.index(sug.section) - 1]
    chk = check_combinations(build(lighter))
    assert not chk.ok
    assert (not chk.strength_ok) if lighter_fails == "ELU" else (chk.strength_ok and not chk.deflection_ok)


def test_suggest_section_combinations_vs_unfactored_fs():
    """q_D = q_L = 10 N/mm + peso propio, L = 6 m (caso de la auditoría QA): el criterio previo
    (FS ≥ 1,5 con D + L sin mayorar ⇒ σ ≤ 0,667·Sy) admite IPE 300, que no verifica ELU
    (1.2D + 1.6L ≈ 1.4·(D + L) ⇒ σ ≤ φ·Sy/1,4 = 0,643·Sy; utilización ≈ 102 %). Con φ la sugerencia
    es IPE 330 y coincide con el veredicto por combinaciones."""
    from core.model import LoadCase
    from core.verification import check_combinations

    span = 6000.0

    def build(sec: Section):
        return build_beam(span, [BeamSupport(0.0, ST.PINNED), BeamSupport(span, ST.ROLLER)], sec, MAT,
                          dist_loads=[BeamDistLoad(0.0, span, -10.0, -10.0),
                                      BeamDistLoad(0.0, span, -10.0, -10.0, case=LoadCase.LIVE)],
                          self_weight=True)

    legacy = suggest_section(build, "IPE", 1.5, deflection_limit_ratio=0.0)
    lrfd = suggest_section(build, "IPE", 1.5, deflection_limit_ratio=0.0, phi=0.9)
    assert legacy is not None and lrfd is not None
    assert (legacy.section.name, lrfd.section.name) == ("IPE 300", "IPE 330")
    legacy_chk = check_combinations(build(legacy.section), deflection_limit_ratio=0.0)
    assert not legacy_chk.strength_ok and legacy_chk.utilization > 1.0
    with pytest.raises(ValueError, match="φ"):
        suggest_section(build, "IPE", 1.5, phi=0.0)
