"""Corte elástico τ = V·Q/(I·t), von Mises y flexión en eje débil / secciones asimétricas.

Backlog AGENTS.md §9, ítems 4 y 5. Referencias:
- Jourawski / Gere & Goodno, "Mechanics of Materials", §5.8-5.10: τ = V·Q/(I·t); rectángulo
  τ_máx = 1,5·V/A; círculo macizo τ_máx = 4V/(3A); tubo de pared delgada τ_máx ≈ 2V/A.
- von Mises en estado plano σ-τ: σ_vM = √(σ² + 3τ²) (Gere & Goodno §7.4; Shigley §5-5).
- Tablas: Euronorm 19-57 (IPE, Wpl e Iz), DIN 1025-1 (IPN, S_x), DIN 1026-1 (UPN, S_x, Iy, Wy, e),
  AISC Shapes Database v15.0 (W, Zx e Iy en unidades US).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam
from core.design import suggest_section
from core.materials import (
    _IPE,
    _IPN,
    _UPN,
    _W,
    MATERIALS,
    SECTIONS,
    Section,
    chs,
    custom,
    get_section,
    rhs,
    solid_rect,
    solid_round,
)
from core.model import COMB_1_4D, LoadCombination
from core.model import SupportType as ST
from core.solver import solve
from core.verification import (
    Status,
    check_combinations,
    check_safety,
    member_stresses,
    shear_stress,
    von_mises,
    weak_axis_bending_stress,
)

REL = 1e-6  # solver exacto (AGENTS.md §7.1)
_TAB_REL = 0.01  # contra tablas de catálogo: redondeo de tabla y acuerdos (< 1 %)
STEEL = MATERIALS["ASTM A36"]  # Sy = 250 MPa
_IN3, _IN4 = 16_387.064, 416_231.426  # in³ -> mm³, in⁴ -> mm⁴


# --------------------------------------------------------------------------- #
# Propiedades de corte y de eje débil del catálogo
# --------------------------------------------------------------------------- #

# Q_max de tabla [mm³]: doble T doblemente simétrico Q = Wpl/2 (Zx/2); IPN y UPN: S_x de DIN.
_Q_TABLE = {
    "IPE 200": 220.6e3 / 2,  # Wpl,y = 220,6 cm³
    "IPE 300": 628.4e3 / 2,  # Wpl,y = 628,4 cm³
    "IPE 600": 3512e3 / 2,  # Wpl,y = 3512 cm³
    "IPN 200": 125e3,  # S_x = 125 cm³
    "UPN 100": 24.5e3,  # S_x = 24,5 cm³
    "UPN 200": 114e3,  # S_x = 114 cm³
    "W 310x38.7": 37.2 * _IN3 / 2,  # W12x26: Zx = 37,2 in³
    "W 200x22.5": 13.6 * _IN3 / 2,  # W8x15: Zx = 13,6 in³
}

# Inercia de eje débil de tabla [mm⁴] (en IPE y W se calcula: control de la geometría).
_IZ_TABLE = {
    "IPE 80": 8.49e4,
    "IPE 200": 142.4e4,
    "IPE 300": 603.8e4,
    "IPE 600": 3387e4,
    "W 310x38.7": 17.3 * _IN4,  # W12x26
    "W 200x22.5": 3.41 * _IN4,  # W8x15
    "W 150x13": 1.99 * _IN4,  # W6x8.5
}

# Módulo resistente de eje débil de tabla [mm³] (en UPN = Iy / (b − e): controla e).
_WY_TABLE = {"IPE 200": 28.47e3, "IPN 200": 26.0e3, "UPN 100": 8.49e3, "UPN 200": 27.0e3,
             "UPN 300": 67.8e3}


@pytest.mark.parametrize("name,Q", _Q_TABLE.items())
def test_catalog_Q_max_matches_table(name, Q):
    """Q_max geométrico (con acuerdos en IPE/W, tf medio en IPN/UPN) = Wpl/2 o S_x de tabla."""
    assert get_section(name).Q_max == pytest.approx(Q, rel=_TAB_REL)


@pytest.mark.parametrize("name,Iz", _IZ_TABLE.items())
def test_catalog_weak_axis_inertia_matches_table(name, Iz):
    """Iz = 2·tf·b³/12 + (h − 2tf)·tw³/12 + 4 acuerdos (r deducido del área) = Iz de tabla."""
    assert get_section(name).Iy == pytest.approx(Iz, rel=_TAB_REL)


@pytest.mark.parametrize("name,Wy", _WY_TABLE.items())
def test_catalog_weak_axis_modulus_matches_table(name, Wy):
    """Wy = Iy / c_y,máx; en UPN c_y,máx = b − e (puntas de ala)."""
    assert get_section(name).Wy == pytest.approx(Wy, rel=_TAB_REL)


def test_rolled_tables_tw_tf_transcription():
    """Control de transcripción de tw y tf contra el área de tabla A.

    A_rect = 2·b·tf + (h − 2tf)·tw. Alas paralelas (IPE, W): A − A_rect = 4·(1 − π/4)·r² ≥ 0 y
    pequeño (< 6 % de A). Alas inclinadas con tf medio (IPN, UPN): A_rect ≈ A (< 2 %).
    """
    for h, b, A, _I, tw, tf in _IPE.values():
        fillets = A * 100.0 - (2 * b * tf + (h - 2 * tf) * tw)
        assert 0.0 < fillets < 0.06 * A * 100.0
    for d, bf, A, _I, tw, tf in _W.values():
        fillets = A - (2 * bf * tf + (d - 2 * tf) * tw)
        assert 0.0 < fillets < 0.06 * A
    for h, b, A, _I, tw, tf, *_ in (*_IPN.values(), *_UPN.values()):
        assert 2 * b * tf + (h - 2 * tf) * tw == pytest.approx(A * 100.0, rel=0.02)


def test_every_catalog_section_has_shear_and_weak_axis_data():
    """Todas las secciones de catálogo: Q, t, Iy, Wy, c_y > 0; en doble T y U, Iy < I."""
    for family, secs in SECTIONS.items():
        for s in secs.values():
            assert s.has_shear_data and s.Q_max > 0 and s.t_web > 0, s.name
            assert s.Iy > 0 and s.Wy > 0 and s.c_y1 > 0 and s.c_y2 > 0, s.name
            assert s.Q_y > 0 and s.t_y > 0, s.name
            assert s.Wy == pytest.approx(s.Iy / s.c_y_max, rel=1e-12)
            if family in ("IPE", "IPN", "UPN", "Perfil W", "Perfil C (Conformado)"):
                assert s.Iy < s.I, s.name


def test_upn_weak_axis_is_asymmetric():
    """UPN 200 (DIN 1026-1): e = 2,01 cm al dorso del alma ⇒ c_y1 = 20,1 mm ≠ c_y2 = b − e = 54,9 mm."""
    s = get_section("UPN 200")
    assert s.c_y1 == pytest.approx(20.1) and s.c_y2 == pytest.approx(75.0 - 20.1)


def test_cold_formed_c_weak_axis_thin_wall_line_method():
    """C 100x50x2: método lineal (línea media, esquinas vivas): x̄ y Iy contra la integración exacta.

    Línea media: alas b' = B − t/2, alma a' = H − t; x̄ = 2·b'·t·(b'/2)/A' + t/2 desde el dorso;
    Iy = 2·t·b'³/12 + 2·t·b'·(b'/2 − x_c)² + a'·t·x_c², x_c = x̄ − t/2. Diferencia < 3 % (radios).
    """
    H, B, t = 100.0, 50.0, 2.0
    s = get_section("C 100x50x2")
    bl, al = B - t / 2, H - t
    A_l = t * (2 * bl + al)
    x_c = 2 * bl * t * (bl / 2) / A_l
    Iy = 2 * t * bl**3 / 12 + 2 * t * bl * (bl / 2 - x_c) ** 2 + al * t * x_c**2
    assert s.c_y1 == pytest.approx(x_c + t / 2, rel=0.03)
    assert s.Iy == pytest.approx(Iy, rel=0.03)
    assert s.Q_y == pytest.approx(t * (B - s.c_y1) ** 2, rel=1e-12)  # dos alas, lado de las puntas


# --------------------------------------------------------------------------- #
# Fórmulas cerradas de τ por forma de sección
# --------------------------------------------------------------------------- #


def test_shear_rectangle_is_1_5_V_over_A():
    """Rectángulo: Q = b·h²/8, t = b ⇒ τ_máx = 1,5·V/A (y en eje débil también)."""
    s, V = solid_rect("50x120", 50.0, 120.0), 30_000.0
    assert shear_stress(s, V) == pytest.approx(1.5 * V / s.A, rel=1e-12)
    assert shear_stress(s.weak_axis(), V) == pytest.approx(1.5 * V / s.A, rel=1e-12)


def test_shear_solid_round_is_4V_over_3A():
    """Círculo macizo: Q = d³/12, t = d ⇒ τ_máx = 4V/(3A)."""
    s, V = solid_round("Ø40", 40.0), 10_000.0
    assert shear_stress(s, V) == pytest.approx(4.0 * V / (3.0 * s.A), rel=1e-12)


def test_shear_thin_tube_tends_to_2V_over_A():
    """Tubo circular de pared delgada: τ_máx → 2V/A (D/t = 100: error < 1 %)."""
    s, V = chs("Ø500x5", 500.0, 5.0), 50_000.0
    assert shear_stress(s, V) == pytest.approx(2.0 * V / s.A, rel=0.01)


def test_shear_rhs_closed_form():
    """Tubo 120x60x4 (H = 120): Q = [B·H² − (B−2t)(H−2t)²]/8, t = 2·4 ⇒ τ = V·Q/(I·2t)."""
    B, H, t, V = 60.0, 120.0, 4.0, 20_000.0
    s = rhs("t", B, H, t)
    Q = (B * H**2 - (B - 2 * t) * (H - 2 * t) ** 2) / 8
    I = (B * H**3 - (B - 2 * t) * (H - 2 * t) ** 3) / 12
    assert shear_stress(s, -V) == pytest.approx(V * Q / (I * 2 * t), rel=1e-12)


def test_shear_i_shape_weak_axis_b2_over_8Iz():
    """Doble T en eje débil: el corte lo toman las dos alas, Q = tf·b²/4, t = 2tf ⇒ τ = V·b²/(8·Iz)."""
    s, V = get_section("IPE 200"), 10_000.0
    assert shear_stress(s.weak_axis(), V) == pytest.approx(V * 100.0**2 / (8.0 * s.Iy), rel=1e-12)


def test_von_mises_closed_form():
    """σ_vM = √(σ² + 3τ²); corte puro: σ_vM = √3·τ."""
    assert von_mises(120.0, 40.0) == pytest.approx(math.sqrt(120.0**2 + 3 * 40.0**2), rel=1e-12)
    assert von_mises(0.0, 50.0) == pytest.approx(math.sqrt(3.0) * 50.0, rel=1e-12)
    assert von_mises(-80.0, 0.0) == pytest.approx(80.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Test 1 — Corte analítico en viga simplemente apoyada IPE 200
# --------------------------------------------------------------------------- #


def _ss_beam(section: Section, q: float, L: float):
    m = build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], section, STEEL,
                   dist_loads=[BeamDistLoad(0, L, q, q)])
    return m, solve(m)


def test_simply_supported_ipe200_tau_at_supports():
    """Viga simple IPE 200, L = 6 m, q = 10 N/mm: V = qL/2 = 30 kN en apoyos.

    τ_máx = V·Q/(I·tw) = 30 000·110 300/(19,43e6·5,6) ≈ 30,4 MPa (Q = Wpl/2 de tabla), en x = 0 y L;
    en el centro V = 0 ⇒ τ = 0.
    """
    L, q = 6000.0, -10.0
    s = get_section("IPE 200")
    m, r = _ss_beam(s, q, L)
    V = abs(q) * L / 2
    chk = check_safety(m, r)
    assert chk.tau_max == pytest.approx(V * s.Q_max / (s.I * s.t_web), rel=REL)
    assert chk.tau_max == pytest.approx(V * _Q_TABLE["IPE 200"] / (s.I * 5.6), rel=_TAB_REL)
    assert chk.tau_max == pytest.approx(30.4, abs=0.1)

    st = member_stresses(s, r.members[0])
    k = int(np.argmax(st.tau))
    assert st.x[k] in (pytest.approx(0.0), pytest.approx(L))
    mid = int(np.argmin(np.abs(st.x - L / 2)))
    assert st.tau[mid] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(st.tau, np.abs(r.members[0].V) * s.Q_max / (s.I * s.t_web), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Test 2 — von Mises en el punto crítico (corte alto + axil)
# --------------------------------------------------------------------------- #

_P, _N, _L_SHORT = 100e3, 200e3, 150.0  # [N], [N], [mm]


def _short_cantilever():
    """Ménsula corta IPE 200, L = 150 mm, extremo libre con P = 100 kN ↓ y N = 200 kN de tracción."""
    s = get_section("IPE 200")
    m = build_beam(_L_SHORT, [BeamSupport(0, ST.FIXED)], s, STEEL,
                   point_loads=[BeamPointLoad(_L_SHORT, Fy=-_P, Fx=_N)])
    return s, m, solve(m)


def test_von_mises_governs_short_cantilever():
    """Empotramiento: N/A = 70,2 MPa, M = P·L = 15 kN·m ⇒ σ_flex = N/A + M·c/I = 147,4 MPa;
    τ = P·Q/(I·tw) = 101,4 MPa ⇒ σ_vM,corte = √((N/A)² + 3τ²) = 189,1 MPa > σ_flex: gobierna el
    corte y FS = Sy/σ_vM = 1,32 < 1,5 (ALERTA)."""
    s, m, r = _short_cantilever()
    chk = check_safety(m, r)
    sN = _N / s.A
    tau = _P * s.Q_max / (s.I * s.t_web)
    sigma_flex = sN + _P * _L_SHORT * s.c_top / s.I
    svm = math.sqrt(sN**2 + 3 * tau**2)

    assert chk.tau_max == pytest.approx(tau, rel=REL)
    assert chk.sigma_max == pytest.approx(sigma_flex, rel=REL)
    assert chk.sigma_vm_max == pytest.approx(svm, rel=REL)
    assert chk.sigma_vm_max > chk.sigma_max and chk.shear_governs
    assert chk.fs == pytest.approx(STEEL.Sy / svm, rel=REL)
    assert chk.utilization == pytest.approx(svm * 1.5 / STEEL.Sy, rel=REL)
    assert chk.status is Status.ALERT
    assert any("σvM" in msg and "τ máx" in msg for msg in chk.issues)

    st = member_stresses(s, r.members[0])
    assert np.allclose(st.sigma_vm, np.maximum(st.sigma_max, np.sqrt(st.sigma_axial**2 + 3 * st.tau**2)))
    assert st.sigma_vm_flex[0] == pytest.approx(sigma_flex, rel=REL)  # fibra extrema
    assert st.sigma_vm_shear[0] == pytest.approx(svm, rel=REL)  # eje neutro


def test_von_mises_in_load_combinations():
    """ELU 1.4D: N y V escalan con γ ⇒ σ_vM,u = 1,4·σ_vM = 264,7 MPa > Sy (FLUENCIA por corte)."""
    s, m, _ = _short_cantilever()
    svm = math.sqrt((_N / s.A) ** 2 + 3 * (_P * s.Q_max / (s.I * s.t_web)) ** 2)
    comb = check_combinations(m, uls_combinations=(COMB_1_4D,), deflection_limit_ratio=0.0)
    assert comb.sigma_vm_max == pytest.approx(1.4 * svm, rel=REL)
    assert comb.tau_max == pytest.approx(1.4 * _P * s.Q_max / (s.I * s.t_web), rel=REL)
    assert comb.utilization == pytest.approx(1.4 * svm / (0.9 * STEEL.Sy), rel=REL)
    assert comb.verdict == (Status.FAIL, "FLUENCIA")
    assert any("σvM,u" in msg for msg in comb.issues)


# --------------------------------------------------------------------------- #
# Test 3 — Secciones asimétricas (c_top ≠ c_bot) y flexión en eje débil
# --------------------------------------------------------------------------- #


def test_asymmetric_section_top_bottom_stresses():
    """Sección T genérica (c_top = 30, c_bot = 70 mm) con N = 50 kN y q ↓ en viga simple L = 4 m.

    Centro: M = qL²/8 > 0 ⇒ σ_top = N/A − M·c_top/I (compresión), σ_bot = N/A + M·c_bot/I (tracción),
    |σ_bot| > |σ_top|; σ_max = σ_bot.
    """
    A, I, L, q, N = 3000.0, 4.0e6, 4000.0, -2.0, 50e3
    tee = Section("T", "Personalizado", A=A, I=I, c_top=30.0, c_bot=70.0, h=100.0, b=120.0)
    m = build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], tee, STEEL,
                   point_loads=[BeamPointLoad(L, Fx=N)], dist_loads=[BeamDistLoad(0, L, q, q)])
    mr = solve(m).members[-1]  # el nodo de la carga axil es el último; centro en la última barra
    mid = int(np.argmin(np.abs(mr.points[:, 0] - L / 2)))
    M = abs(q) * L**2 / 8
    assert mr.M[mid] == pytest.approx(M, rel=REL)
    assert mr.sigma_top[mid] == pytest.approx(N / A - M * 30.0 / I, rel=REL)
    assert mr.sigma_bot[mid] == pytest.approx(N / A + M * 70.0 / I, rel=REL)
    assert abs(mr.sigma_top[mid]) != pytest.approx(abs(mr.sigma_bot[mid]), rel=0.05)
    assert tee.W_top == pytest.approx(I / 30.0) and tee.W_bot == pytest.approx(I / 70.0)
    assert tee.W == tee.W_bot  # módulo mínimo = fibra más alejada


@pytest.mark.parametrize("flip", [False, True])
def test_upn_weak_axis_beam_asymmetric_fibers(flip):
    """UPN 200 en eje débil, viga simple L = 3 m, q = 2 N/mm: M = qL²/8 = 2,25 kN·m.

    Sin invertir: fibra superior = dorso del alma (c_top = e = 20,1 mm), inferior = puntas (54,9 mm).
    σ_top = −M·c_top/Iy, σ_bot = +M·c_bot/Iy (σ_top ≠ σ_bot); σ_max = M/Wy = weak_axis_bending_stress.
    """
    L, q = 3000.0, -2.0
    s = get_section("UPN 200")
    w = s.weak_axis(flip=flip)
    c_top, c_bot = (s.c_y2, s.c_y1) if flip else (s.c_y1, s.c_y2)
    assert (w.I, w.c_top, w.c_bot) == (s.Iy, c_top, c_bot)
    m, r = _ss_beam(w, q, L)
    mr = r.members[0]
    mid = int(np.argmin(np.abs(mr.x - L / 2)))
    M = abs(q) * L**2 / 8
    assert mr.sigma_top[mid] == pytest.approx(-M * c_top / s.Iy, rel=REL)
    assert mr.sigma_bot[mid] == pytest.approx(M * c_bot / s.Iy, rel=REL)
    assert mr.sigma_bot[mid] / mr.sigma_top[mid] == pytest.approx(-c_bot / c_top, rel=REL)
    chk = check_safety(m, r, deflection_limit_ratio=0.0)
    assert chk.sigma_max == pytest.approx(M / s.Wy, rel=REL)
    assert chk.sigma_max == pytest.approx(weak_axis_bending_stress(s, M), rel=REL)
    assert not chk.shear_governs
    # El corte en eje débil usa Q_y y t_y (alas): τ = V·tf·(b − e)²/(Iy·2tf) = V·(b − e)²/(2·Iy)
    assert chk.tau_max == pytest.approx(abs(q) * L / 2 * s.c_y2**2 / (2 * s.Iy), rel=REL)


def test_weak_axis_bending_stress_and_strong_vs_weak():
    """σ_y = M_y/W_y; para el mismo M, el IPE 200 en eje débil tensiona W/Wy ≈ 6,8 veces más."""
    s, M = get_section("IPE 200"), 5e6
    assert weak_axis_bending_stress(s, -M) == pytest.approx(M / s.Wy, rel=1e-12)
    assert weak_axis_bending_stress(s, M) / (M / s.W) == pytest.approx(s.W / s.Wy, rel=1e-12)
    assert s.W / s.Wy == pytest.approx(194.3 / 28.47, rel=_TAB_REL)


def test_weak_axis_roundtrip_and_errors():
    """weak_axis() intercambia ejes y es involutiva; sin datos de eje débil lanza ValueError."""
    names = ("IPE 200", "UPN 200", "W 200x22.5", "C 100x50x2", "Tubo 120x60x4",
             'Caño 4" Sch40 (Ø114.3x6.02)')
    for name in names:
        s = get_section(name)
        w = s.weak_axis()
        assert (w.I, w.Iy, w.Q_max, w.Q_y, w.t_web, w.t_y) == (s.Iy, s.I, s.Q_y, s.Q_max, s.t_y, s.t_web)
        assert w.Wy == pytest.approx(s.W, rel=1e-12) and w.W == pytest.approx(s.Wy, rel=1e-12)
        assert w.A == s.A and w.family == s.family and w.name.endswith("(eje débil)")
        assert w.weak_axis() == s
    with pytest.raises(ValueError, match="eje débil"):
        custom("x", A=1000.0, I=1e6, c=50.0).weak_axis()
    with pytest.raises(ValueError, match="eje débil"):
        weak_axis_bending_stress(custom("x", A=1000.0, I=1e6, c=50.0), 1e6)
    with pytest.raises(ValueError, match="≥ 0"):
        Section("x", "Personalizado", A=1.0, I=1.0, c_top=1.0, c_bot=1.0, h=2.0, Q_max=-1.0)


def test_suggest_section_weak_axis_needs_heavier_profile():
    """suggest_section con el perfil orientado en eje débil (build ← s.weak_axis()).

    Viga simple L = 3 m, q = 2 N/mm: M = 2,25e6 N·mm; con FS 1,5: W_req = M·1,5/250 = 13,5 cm³.
    Eje fuerte: IPE 80 (W = 20,0 cm³). Eje débil: Wy ≥ 13,5 cm³ ⇒ IPE 160 (Wy = 16,7 cm³; IPE 140: 12,3).
    """
    L, q = 3000.0, -2.0

    def build(sec: Section):
        return build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], sec, STEEL,
                          dist_loads=[BeamDistLoad(0, L, q, q)])

    strong = suggest_section(build, "IPE", fs_min=1.5, deflection_limit_ratio=0.0)
    weak = suggest_section(lambda sec: build(sec.weak_axis()), "IPE", fs_min=1.5,
                           deflection_limit_ratio=0.0)
    assert strong is not None and strong.section.name == "IPE 80"
    assert weak is not None and weak.section.name == "IPE 160"
    assert weak.check.sigma_max == pytest.approx(abs(q) * L**2 / 8 / weak.section.Wy, rel=REL)


# --------------------------------------------------------------------------- #
# Test 4 — Retrocompatibilidad: flexión dominante y secciones sin datos de corte
# --------------------------------------------------------------------------- #


def test_bending_dominated_results_unchanged():
    """Viga simple IPE 200, L = 6 m: σ_flex ≫ √3·τ ⇒ σ_vM = σ_max y FS = Sy/σ_max (bit a bit)."""
    m, r = _ss_beam(get_section("IPE 200"), -10.0, 6000.0)
    chk = check_safety(m, r)
    assert chk.sigma_vm_max == chk.sigma_max
    assert chk.fs == STEEL.Sy / chk.sigma_max
    assert chk.utilization == chk.sigma_max * 1.5 / STEEL.Sy
    assert chk.x == pytest.approx(3000.0) and not chk.shear_governs
    assert chk.issues[0].startswith("Resistencia: FS = ") and "(σ máx = " in chk.issues[0]


def test_bending_dominated_combination_messages_unchanged():
    """Combinaciones con flexión dominante: el texto de ELU sigue siendo 'σu = … MPa'."""
    m, _ = _ss_beam(get_section("IPE 200"), -10.0, 6000.0)
    comb = check_combinations(m, uls_combinations=(LoadCombination("1.4D", 1.4, 0.0),),
                              deflection_limit_ratio=0.0)
    assert comb.sigma_vm_max == comb.strength.sigma_max
    assert any(msg.startswith("ELU 1.4D: σu = ") for msg in comb.issues)


def test_custom_section_without_shear_data_is_pure_bending():
    """Sección personalizada sin Q ni t: τ = 0 y la verificación es la de tensión normal pura,
    aún en la ménsula corta donde con datos de corte gobernaría von Mises."""
    ipe = get_section("IPE 200")
    sec = custom("sin Q", A=ipe.A, I=ipe.I, c=ipe.h / 2)
    assert not sec.has_shear_data and sec.shear_factor == 0.0
    m = build_beam(_L_SHORT, [BeamSupport(0, ST.FIXED)], sec, STEEL,
                   point_loads=[BeamPointLoad(_L_SHORT, Fy=-_P, Fx=_N)])
    chk = check_safety(m, solve(m))
    assert chk.tau_max == 0.0
    assert chk.sigma_vm_max == chk.sigma_max
    assert chk.fs == STEEL.Sy / chk.sigma_max
    with_q = custom("con Q", A=ipe.A, I=ipe.I, c=ipe.h / 2, Q_max=ipe.Q_max, t_web=ipe.t_web)
    assert shear_stress(with_q, _P) == pytest.approx(shear_stress(ipe, _P), rel=1e-12)
