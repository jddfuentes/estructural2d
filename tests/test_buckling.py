"""Pandeo flexional y pandeo lateral-torsional (backlog 3): AISC 360-16 = CIRSOC 301-2005.

Referencias: AISC 360-16 §E3 (compresión), §F2 / §F11 (PLT), §H1.1 (interacción);
Timoshenko y Gere, "Theory of Elastic Stability" (1961) §6.2 (Mcr de doble T);
tablas ArcelorMittal / EN 10365 (IPE) y AISC Shapes Database v15.0 (W) para J, Cw, Z, r_ts.
Unidades SI-mm: N, mm, MPa, N·mm.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import math

import pytest

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam, build_portal_frame
from core.materials import MATERIALS, custom, get_section, rhs, solid_rect, solid_round
from core.model import LoadCase, Member, Model, NodalLoad, Support
from core.model import SupportType as ST
from core.reports import build_pdf_report
from core.solver import solve
from core.verification import (
    BucklingRegime,
    LTBZone,
    Status,
    beam_ltb_capacity,
    check_combinations,
    check_safety,
    column_buckling_capacity,
    interaction_h1,
    physical_members,
)

E = 200_000.0
FY = 250.0
STEEL = MATERIALS["ASTM A36"]  # E = 200 000 MPa, Sy = 250 MPa
IN = 25.4  # [mm/in], sólo para comparar con la tabla AISC en unidades US


# --------------------------------------------------------------------------- #
# Propiedades de estabilidad del catálogo
# --------------------------------------------------------------------------- #


def test_ipe200_torsion_warping_plastic_vs_arcelormittal_table():
    """IPE 200 (ArcelorMittal / EN 10365): It = 6,98 cm⁴, Iw = 12,99·10³ cm⁶, Wpl,y = 220,6 cm³.

    J por El Darwish-Johnston con acuerdos; Cw = Iz·h0²/4; Z = 2·Q_max.
    """
    s = get_section("IPE 200")
    assert s.J == pytest.approx(6.98e4, rel=1.5e-2)
    assert s.Cw == pytest.approx(12.99e9, rel=1.5e-2)
    assert s.Z == pytest.approx(220.6e3, rel=1e-2)
    assert s.h0 == pytest.approx(200.0 - 8.5)
    assert s.r_y == pytest.approx(22.4, rel=1e-2)  # iz de tabla = 2,24 cm


def test_w8x15_properties_vs_aisc_shapes_database():
    """W 200x22.5 = W8x15 (AISC v15.0): J = 0,137 in⁴, Cw = 51,8 in⁶, r_ts = 1,06 in, Zx = 13,6 in³."""
    s = get_section("W 200x22.5")
    assert s.J / IN**4 == pytest.approx(0.137, rel=1.5e-2)
    assert s.Cw / IN**6 == pytest.approx(51.8, rel=1.5e-2)
    assert s.r_ts / IN == pytest.approx(1.06, rel=1.5e-2)
    assert s.Z / IN**3 == pytest.approx(13.6, rel=1.5e-2)


def test_radii_of_gyration_and_rts_identity():
    """r_x = √(I/A), r_y = √(Iy/A), r_min = mín; doble T simétrico: r_ts² = Iy·h0/(2·S) (AISC F2-7, nota)."""
    s = get_section("IPE 300")
    assert s.r_x == pytest.approx(math.sqrt(s.I / s.A), rel=1e-12)
    assert s.r_y == pytest.approx(math.sqrt(s.Iy / s.A), rel=1e-12)
    assert s.r_min == s.r_y
    assert s.r_ts**2 == pytest.approx(s.Iy * s.h0 / (2.0 * s.W), rel=1e-12)
    # Sección personalizada sin Iy: sólo se conoce r_x (pandeo en el plano).
    c = custom("X", A=1000.0, I=2e6, c=50.0)
    assert c.r_min == pytest.approx(math.sqrt(2e3), rel=1e-12)


def test_plastic_modulus_closed_forms_and_weak_axis_roundtrip():
    """Z = 2·Q: rectángulo b·h²/4, redondo d³/6. weak_axis() intercambia Z ↔ Z_y y conserva J, Cw, h0."""
    r = solid_rect("10x100", 10.0, 100.0)
    assert r.Z == pytest.approx(10.0 * 100.0**2 / 4.0, rel=1e-12)
    assert solid_round("Ø40", 40.0).Z == pytest.approx(40.0**3 / 6.0, rel=1e-12)
    for name in ("IPE 200", "UPN 200", "C 200x75x3", "Tubo 120x60x4"):
        s = get_section(name)
        w = s.weak_axis()
        assert (w.Z, w.Z_y, w.J, w.Cw, w.h0) == (s.Z_y, s.Z, s.J, s.Cw, s.h0)
        assert w.weak_axis() == s


# --------------------------------------------------------------------------- #
# Test 1 — Pandeo flexional de columnas (AISC 360-16 §E3)
# --------------------------------------------------------------------------- #


def test_short_column_inelastic_closed_form():
    """Redondo Ø40 (r = d/4 = 10 mm), L = 500 mm: λ = 50 ≤ 4,71·√(E/Fy) = 133,2 → inelástico.

    Fe = π²E/λ² = 789,6 MPa; Fcr = 0,658^(Fy/Fe)·Fy = 219,0 MPa (E3-2, E3-4); Pn = Fcr·A; φPn = 0,9·Pn.
    """
    s = solid_round("Ø40", 40.0)
    chk = column_buckling_capacity(s, 500.0, K=1.0, Fy=FY, E=E, N_comp=-50e3)
    Fe = math.pi**2 * E / 50.0**2
    assert chk.r_min == pytest.approx(10.0, rel=1e-12)
    assert chk.slenderness == pytest.approx(50.0, rel=1e-12)
    assert chk.slenderness_limit == pytest.approx(4.71 * math.sqrt(E / FY), rel=1e-12)
    assert chk.regime is BucklingRegime.INELASTIC
    assert chk.Fe == pytest.approx(Fe, rel=1e-12)
    assert chk.Fcr == pytest.approx(0.658 ** (FY / Fe) * FY, rel=1e-12)
    assert chk.Fcr == pytest.approx(218.97, rel=1e-4)
    assert chk.Pn == pytest.approx(chk.Fcr * s.A, rel=1e-12)
    assert chk.phi_Pn == pytest.approx(0.9 * chk.Pn, rel=1e-12)
    assert chk.utilization == pytest.approx(50e3 / chk.phi_Pn, rel=1e-12)


def test_slender_column_matches_0877_euler():
    """IPE 200, L = 4 m: λ = L/r_y = 179 > 150 → elástico, Fcr = 0,877·π²E/λ² (E3-3)."""
    s = get_section("IPE 200")
    chk = column_buckling_capacity(s, 4000.0, K=1.0, Fy=FY, E=E)
    lam = 4000.0 / s.r_y
    assert lam > 150.0
    assert chk.r_min == s.r_y
    assert chk.regime is BucklingRegime.ELASTIC
    assert chk.Fcr == pytest.approx(0.877 * math.pi**2 * E / lam**2, rel=1e-12)
    assert chk.Fcr < chk.Fe  # 0,877·Fe (curvatura inicial L/1000)


def test_column_transition_k_factor_and_squash_load():
    """Continuidad en λ = 4,71·√(E/Fy) (Fy/Fe = 2,25: 0,658^2,25 = 0,390 ≈ 0,877/2,25 = 0,390);
    K multiplica la esbeltez; L = 0 → Fcr = Fy (aplastamiento)."""
    s = solid_round("Ø40", 40.0)
    L_lim = 4.71 * math.sqrt(E / FY) * 10.0
    a = column_buckling_capacity(s, L_lim * (1.0 - 1e-9), Fy=FY, E=E)
    b = column_buckling_capacity(s, L_lim * (1.0 + 1e-9), Fy=FY, E=E)
    assert (a.regime, b.regime) == (BucklingRegime.INELASTIC, BucklingRegime.ELASTIC)
    assert a.Fcr == pytest.approx(b.Fcr, rel=2e-3)
    assert column_buckling_capacity(s, 800.0, K=2.0).slenderness == pytest.approx(160.0, rel=1e-12)
    assert column_buckling_capacity(s, 800.0, K=2.0).KL == pytest.approx(1600.0)
    assert column_buckling_capacity(s, 0.0, Fy=FY).Fcr == pytest.approx(FY, rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"L": -1.0}, {"K": 0.0}, {"Fy": 0.0}, {"phi_c": 1.1}])
def test_column_invalid_inputs(kwargs):
    args = {"L": 1000.0, "K": 1.0, "Fy": FY, "phi_c": 0.9} | kwargs
    with pytest.raises(ValueError):
        column_buckling_capacity(get_section("IPE 200"), **args)


# --------------------------------------------------------------------------- #
# Test 2 — Pandeo lateral-torsional (AISC 360-16 §F2, §F11)
# --------------------------------------------------------------------------- #


def test_ltb_continuous_bracing_reaches_mp():
    """Lb = 0 (arriostramiento continuo): Mn = Mp = Fy·Z (F2-1)."""
    s = get_section("IPE 200")
    chk = beam_ltb_capacity(s, 0.0, Fy=FY, E=E, M_u=30e6)
    assert chk.zone is LTBZone.PLASTIC
    assert chk.Mn == pytest.approx(FY * s.Z, rel=1e-12)
    assert chk.Mn > chk.My  # factor de forma ≈ 1,14
    assert chk.utilization == pytest.approx(30e6 / (0.9 * FY * s.Z), rel=1e-12)


def test_ltb_long_span_below_my_matches_timoshenko():
    """IPE 200, Lb = 8 m: PLT elástico, Mn = Fcr·S < My y coincide con la solución clásica
    Mcr = (π/Lb)·√(E·Iy·G·J + (π·E/Lb)²·Iy·Cw) (Timoshenko y Gere §6.2), con G = E/2,6 (ν = 0,3).
    AISC F2-4 equivale a G/E = 0,385 (coeficiente 0,078 redondeado): tolerancia 1e-3."""
    s = get_section("IPE 200")
    Lb = 8000.0
    chk = beam_ltb_capacity(s, Lb, Fy=FY, E=E)
    G = E / 2.6
    Mcr = math.pi / Lb * math.sqrt(E * s.Iy * G * s.J + (math.pi * E / Lb) ** 2 * s.Iy * s.Cw)
    assert chk.zone is LTBZone.ELASTIC
    assert chk.Mn < chk.My
    assert chk.Mn == pytest.approx(Mcr, rel=1e-3)
    assert chk.Mn == pytest.approx(chk.Fcr * s.W, rel=1e-12)


def test_ltb_zone_limits_are_continuous():
    """Lp = 1,76·r_y·√(E/Fy) ≈ 1112 mm (IPE 200, F2-5); Mn(Lp) = Mp; Mn(Lr) = 0,7·Fy·S (F2-2) y F2-4 en
    Lr da Fcr = 0,7·Fy (define Lr, F2-6): continuidad entre zonas 2 y 3."""
    s = get_section("IPE 200")
    base = beam_ltb_capacity(s, 0.0, Fy=FY, E=E)
    assert base.Lp == pytest.approx(1.76 * s.r_y * math.sqrt(E / FY), rel=1e-12)
    assert base.Lp == pytest.approx(1112.0, rel=1e-3)
    assert beam_ltb_capacity(s, base.Lp, Fy=FY, E=E).Mn == pytest.approx(base.Mp, rel=1e-12)
    at_lr = beam_ltb_capacity(s, base.Lr, Fy=FY, E=E)
    past_lr = beam_ltb_capacity(s, base.Lr * (1.0 + 1e-12), Fy=FY, E=E)
    assert at_lr.zone is LTBZone.INELASTIC and past_lr.zone is LTBZone.ELASTIC
    assert at_lr.Mn == pytest.approx(0.7 * FY * s.W, rel=1e-9)
    # F2-6 usa constantes redondeadas (1,95 y 6,76): F2-4 en Lr reproduce 0,7·Fy con error < 0,1 %.
    assert past_lr.Mn == pytest.approx(0.7 * FY * s.W, rel=1e-3)
    mns = [beam_ltb_capacity(s, Lb, Fy=FY, E=E).Mn for Lb in range(0, 12001, 500)]
    assert all(a >= b for a, b in zip(mns, mns[1:], strict=False))  # no crece con Lb


def test_ltb_cb_scales_and_is_capped_by_mp():
    """Zona inelástica: Mn = Cb·[...] ≤ Mp (F2-2)."""
    s = get_section("IPE 200")
    m1 = beam_ltb_capacity(s, 3000.0, Fy=FY, E=E, Cb=1.0)
    m13 = beam_ltb_capacity(s, 3000.0, Fy=FY, E=E, Cb=1.1)
    assert m13.Mn == pytest.approx(min(1.1 * m1.Mn, m1.Mp), rel=1e-12)
    assert beam_ltb_capacity(s, 3000.0, Fy=FY, E=E, Cb=3.0).Mn == pytest.approx(m1.Mp, rel=1e-12)
    with pytest.raises(ValueError):
        beam_ltb_capacity(s, 3000.0, Cb=0.9)


def test_ltb_channel_uses_c_factor():
    """UPN y C conformado: F2 con c = (h0/2)·√(Iy/Cw) ≠ 1 (F2-8b); la resistencia baja con Lb."""
    for name in ("UPN 200", "C 200x75x3"):
        s = get_section(name)
        short, long_ = beam_ltb_capacity(s, 0.0), beam_ltb_capacity(s, 6000.0)
        assert short.zone is LTBZone.PLASTIC and long_.zone is LTBZone.ELASTIC
        assert long_.Mn < long_.My
        assert "F2" in long_.method


def test_ltb_solid_rectangular_bar_f11():
    """Planchuela 10 × 100 de canto (F11): Lb = 0 → Mp = 1,5·My (≤ 1,6·My); en Lr = 1,9·E·t²/(Fy·d)
    F11-2 da [1,52 − 0,274·1,9]·My = 0,9994·My y F11-3 da My: continuidad."""
    s = solid_rect("10x100", 10.0, 100.0)
    base = beam_ltb_capacity(s, 0.0, Fy=FY, E=E)
    assert base.zone is LTBZone.PLASTIC and base.Mn == pytest.approx(1.5 * base.My, rel=1e-12)
    Lr = 1.9 * E * 10.0**2 / (FY * 100.0)
    assert base.Lr == pytest.approx(Lr, rel=1e-12)
    assert beam_ltb_capacity(s, Lr, Fy=FY, E=E).Mn == pytest.approx(base.My, rel=1e-3)
    far = beam_ltb_capacity(s, 4.0 * Lr, Fy=FY, E=E)
    assert far.zone is LTBZone.ELASTIC
    assert far.Mn == pytest.approx(1.9 * E / (4.0 * Lr * 100.0 / 10.0**2) * s.W, rel=1e-12)


def test_ltb_not_applicable_or_not_evaluated():
    """Sin PLT (Mn = Mp sin importar Lb): eje débil, tubos, redondos. Personalizada sin J/Cw: no evaluado."""
    weak = get_section("IPE 200").weak_axis()
    chk = beam_ltb_capacity(weak, 10_000.0, Fy=FY)
    assert chk.zone is LTBZone.NOT_APPLICABLE
    assert chk.Mn == pytest.approx(min(FY * weak.Z, 1.6 * FY * weak.W), rel=1e-12)
    for s in (rhs("Tubo", 60.0, 120.0, 4.0), solid_round("Ø40", 40.0), get_section("Tubo 100x100x5")):
        assert beam_ltb_capacity(s, 10_000.0).zone is LTBZone.NOT_APPLICABLE
    nd = beam_ltb_capacity(custom("X", A=2000.0, I=1e7, c=100.0), 10_000.0, Fy=FY)
    assert nd.zone is LTBZone.NOT_EVALUATED
    assert nd.Mn == pytest.approx(FY * 1e7 / 100.0, rel=1e-12)  # sin Z ni Q: Mp = My (conservador)


def test_interaction_h1():
    """AISC 360-16 H1-1a (Pr/Pc ≥ 0,2): 0,5 + 8/9·0,45 = 0,9; H1-1b: 0,1/2 + 0,5 = 0,55."""
    assert interaction_h1(50.0, 100.0, 45.0, 100.0) == pytest.approx(0.9, rel=1e-12)
    assert interaction_h1(-10.0, 100.0, -50.0, 100.0) == pytest.approx(0.55, rel=1e-12)


# --------------------------------------------------------------------------- #
# Integración en check_safety / check_combinations
# --------------------------------------------------------------------------- #


def _ss_beam(L: float = 8000.0, q: float = -1.0, point: bool = False) -> Model:
    loads = [BeamPointLoad(L / 2.0, Fy=-1.0)] if point else []
    return build_beam(L, [BeamSupport(0.0, ST.PINNED), BeamSupport(L, ST.ROLLER)],
                      get_section("IPE 200"), STEEL, point_loads=loads,
                      dist_loads=[BeamDistLoad(0.0, L, q, q)])


def _cantilever_column(P: float, L: float = 3000.0) -> Model:
    return Model(nodes=[(0.0, 0.0), (0.0, L)],
                 members=[Member(0, 1, get_section("IPE 200"), STEEL)],
                 supports=[Support(0, ST.FIXED)],
                 nodal_loads=[NodalLoad(1, Fy=-P)])


def test_physical_members_join_split_spans():
    """La carga puntual parte la viga en 2 barras, pero la barra física es una sola (L = 8 m); un apoyo
    intermedio corta la cadena; en el pórtico, viga partida por cargas + 2 columnas = 3 barras físicas."""
    m = _ss_beam(point=True)
    assert len(m.members) == 2 and physical_members(m) == ((0, 1),)
    cont = build_beam(8000.0, [BeamSupport(0.0, ST.PINNED), BeamSupport(4000.0, ST.ROLLER),
                               BeamSupport(8000.0, ST.ROLLER)], get_section("IPE 200"), STEEL)
    assert physical_members(cont) == ((0,), (1,))
    pts = [BeamPointLoad(2000.0, Fy=-1e3), BeamPointLoad(5000.0, Fy=-1e3)]
    frame = build_portal_frame(8000.0, 4000.0, get_section("IPE 240"), get_section("IPE 300"), STEEL,
                               beam_point_loads=pts)
    chains = physical_members(frame)
    assert len(chains) == 3 and sum(len(c) for c in chains) == len(frame.members)


def test_long_unbraced_beam_fails_ltb_although_stresses_pass():
    """IPE 200, L = 8 m, q tal que M = My/2 (FS = 2 por tensiones): Mn(8 m) = 15,9 kN·m < M = 24,3 kN·m
    → inestable (FLUENCIA de severidad, texto 'Inestabilidad'). Lb = 1 m → Mn = Mp, verifica."""
    s = get_section("IPE 200")
    L = 8000.0
    q = -8.0 * (FY * s.W / 2.0) / L**2
    m = _ss_beam(L, q, point=True)  # carga puntual de 1 N: barra partida en el centro
    r = solve(m)
    chk = check_safety(m, r, deflection_limit_ratio=0.0, stability=True)
    cs = chk.critical_stability
    assert cs is not None and cs.members == (0, 1)
    assert cs.length == pytest.approx(L) and cs.ltb.Lb == pytest.approx(L)
    assert chk.fs == pytest.approx(2.0, rel=1e-3) and chk.strength_ok
    assert cs.M_max == pytest.approx(FY * s.W / 2.0, rel=1e-3)
    assert cs.utilization_nominal == pytest.approx(cs.M_max / cs.ltb.Mn, rel=1e-6)
    assert chk.buckling_failure and not chk.stability_ok and chk.stability_governs
    assert chk.status is Status.FAIL
    text = " ".join(chk.issues)
    assert "Inestabilidad: pandeo lateral-torsional" in text and "Gobierna la inestabilidad" in text
    braced = check_safety(m, r, deflection_limit_ratio=0.0, stability=True, unbraced_length=1000.0)
    assert braced.critical_stability is not None
    assert braced.critical_stability.ltb.zone is LTBZone.PLASTIC
    assert braced.status is Status.OK and braced.issues == ()


def test_cantilever_column_flexural_buckling_k2():
    """Ménsula vertical IPE 200, L = 3 m, K = 2: KL/r = 6000/22,35 = 268,5 (elástico). P = 0,8·Pn:
    utilización P/(Pn/FS) = 0,8·1,5 = 1,2 (H1-1a con M = 0) → ALERTA; nominal 0,8 → no inestable."""
    pn = column_buckling_capacity(get_section("IPE 200"), 3000.0, K=2.0, Fy=FY).Pn
    m = _cantilever_column(0.8 * pn)
    chk = check_safety(m, solve(m), stability=True, effective_length_factor_k=2.0)
    cs = chk.critical_stability
    assert cs is not None
    assert cs.column.KL == pytest.approx(6000.0) and cs.column.regime is BucklingRegime.ELASTIC
    assert cs.N_comp == pytest.approx(0.8 * pn, rel=1e-9)
    assert cs.utilization == pytest.approx(1.2, rel=1e-9)
    assert cs.utilization_nominal == pytest.approx(0.8, rel=1e-9)
    assert cs.mode == "pandeo flexional"
    assert chk.status is Status.ALERT and chk.stability_governs and chk.fs > 10.0
    assert "Estabilidad: pandeo flexional" in chk.issues[0]
    lam = 6000.0 / get_section("IPE 200").r_y
    assert lam == pytest.approx(268.5, rel=1e-3)
    assert any(f"KL/r = {lam:.0f} > 200" in n for n in chk.stability_notes)  # AISC 360-16 §E2


def test_combinations_stability_no_verifica():
    """Misma ménsula con P = 0,7·Pn (D): 1,4D → Pu = 0,98·Pn > φc·Pn = 0,9·Pn → NO VERIFICA en ELU
    aunque σu ≪ φ·Sy; gobierna 1,4D."""
    pn = column_buckling_capacity(get_section("IPE 200"), 3000.0, K=2.0, Fy=FY).Pn
    m = _cantilever_column(0.7 * pn)
    chk = check_combinations(m, stability=True, effective_length_factor_k=2.0)
    assert chk.strength_ok and not chk.stability_ok
    assert chk.stability_utilization == pytest.approx(0.98 / 0.9, rel=1e-9)
    assert chk.governing.name == chk.stability_governing[0].name == "1.4D"
    assert chk.verdict == (Status.FAIL, "NO VERIFICA")
    assert any(i.startswith("ELU 1.4D: Estabilidad") and "φ·Pn" in i for i in chk.issues)


def test_combinations_yield_not_hidden_by_stability_governing_t4b():
    """Auditoría v0.1.0, B2 (caso T4b). Ménsula vertical IPE 200, L = 4 m, K = 2, D = 225 kN axial y
    L = 5 kN lateral en la punta.

    Resistencia, en el empotramiento: σ_u = N_u/A + M_u·c/I con N_u = γD·D y M_u = γL·H·L.
      1.4D:        σ_u = 1,4·225 000/2850 = 110,5 MPa  → 49 % de φ·Sy.
      1.2D + 1.6L: σ_u = 270 000/2850 + 1,6·5000·4000·100/1943·10⁴ = 94,7 + 164,7 = 259,4 MPa > Sy = 250
                   → FS = 0,96: FLUENCIA.
    Estabilidad (KL/r = 8000/22,35 = 358, elástico): la H1 máxima es la de 1.4D (Pu mayor). La envolvente
    de tensiones (`governing`) y la combinación crítica de pandeo (`stability_governing`) son distintas y
    el veredicto debe ser FLUENCIA, no "NO VERIFICA".
    """
    s = get_section("IPE 200")
    L, D, H = 4000.0, 225_000.0, 5000.0
    m = Model(nodes=[(0.0, 0.0), (0.0, L)], members=[Member(0, 1, s, STEEL)],
              supports=[Support(0, ST.FIXED)],
              nodal_loads=[NodalLoad(1, Fy=-D), NodalLoad(1, Fx=H, case=LoadCase.LIVE)])
    chk = check_combinations(m, stability=True, effective_length_factor_k=2.0)

    sigma_u = 1.2 * D / s.A + 1.6 * H * L * s.c_top / s.I
    assert sigma_u == pytest.approx(259.4, abs=0.05)
    by_name = {comb.name: c for comb, c in chk.strength_checks}
    assert by_name["1.2D + 1.6L"].sigma_max == pytest.approx(sigma_u, rel=1e-6)
    fs_u = by_name["1.2D + 1.6L"].fs
    assert fs_u == pytest.approx(FY / sigma_u, rel=1e-6) and fs_u < 1.0
    assert by_name["1.4D"].sigma_max == pytest.approx(1.4 * D / s.A, rel=1e-6)

    # Desacople de criterios: tensiones → 1.2D + 1.6L; estabilidad (H1) → 1.4D
    assert chk.governing.name == "1.2D + 1.6L"
    assert chk.utilization == pytest.approx(max(c.utilization for c in by_name.values()), rel=1e-12)
    assert chk.utilization == pytest.approx(sigma_u / (0.9 * FY), rel=1e-6)
    assert chk.stability_governing[0].name == "1.4D"
    assert chk.stability_utilization == max(c.stability_utilization for c in by_name.values())
    assert not chk.stability_ok and chk.stability_governs
    assert chk.governing_results is chk.uls_results[1]

    # Veredicto: fluencia en cualquier combinación ⇒ FLUENCIA (y status FAIL), aunque la H1 gobierne
    assert chk.yield_failure and chk.strength_failure
    assert chk.status is Status.FAIL
    assert chk.verdict == (Status.FAIL, "FLUENCIA")
    # issues: un mensaje por combinación y criterio
    assert chk.issues[0].startswith("ELU 1.2D + 1.6L: σu = 259.4 MPa supera Sy = 250 MPa (fluencia)")
    assert any(i.startswith("ELU 1.4D: Inestabilidad: pandeo flexional") for i in chk.issues)
    assert not any(i.startswith("ELU 1.4D: σu") for i in chk.issues)  # 1.4D verifica por tensiones


@pytest.mark.parametrize("kwargs", [{"fs_min": 0.9}, {"effective_length_factor_k": 0.0},
                                    {"unbraced_length": -1.0}])
def test_stability_invalid_inputs(kwargs):
    m = _ss_beam()
    with pytest.raises(ValueError):
        check_safety(m, solve(m), stability=True, **kwargs)


# --------------------------------------------------------------------------- #
# Test 3 — Retrocompatibilidad
# --------------------------------------------------------------------------- #


def test_defaults_are_backward_compatible():
    """Sin `stability=True` nada cambia: viga simple IPE 200, L = 6 m, q = 10 kN/m →
    M = qL²/8 = 45 kN·m, FS = Sy·W/M; los parámetros de pandeo se ignoran y no hay excepciones."""
    s = get_section("IPE 200")
    m = _ss_beam(6000.0, -10.0)
    r = solve(m)
    base = check_safety(m, r)
    assert base.stability == () and base.stability_ok and not base.stability_governs
    assert base.stability_utilization == 0.0 and base.critical_stability is None
    assert base.fs == pytest.approx(FY * s.W / (10.0 * 6000.0**2 / 8.0), rel=1e-6)
    assert base == check_safety(m, r, stability=False, unbraced_length=1000.0,
                                effective_length_factor_k=2.0)
    comb = check_combinations(m)
    assert comb.stability_utilization == 0.0 and comb.stability_ok
    assert all(c.stability == () for _, c in comb.strength_checks)
    # Las propiedades nuevas de Section no alteran las de siempre (tabla IPE 200).
    assert (s.A, s.I) == (2850.0, 1943e4)
    assert dataclasses.replace(s, Z=0.0, J=0.0, Cw=0.0, h0=0.0).W == s.W


# --------------------------------------------------------------------------- #
# Memoria PDF: el veredicto no dice "FLUENCIA" cuando falla por inestabilidad
# --------------------------------------------------------------------------- #


def test_pdf_verdict_reports_instability():
    pypdf = pytest.importorskip("pypdf")
    s = get_section("IPE 200")
    m = _ss_beam(8000.0, -8.0 * (FY * s.W / 2.0) / 8000.0**2)
    r = solve(m)
    chk = check_safety(m, r, deflection_limit_ratio=0.0, stability=True)
    pdf = build_pdf_report(m, r, chk, date=dt.date(2026, 10, 11), include_figures=False)
    text = " ".join(" ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages).split())
    assert "INESTABILIDAD" in text
