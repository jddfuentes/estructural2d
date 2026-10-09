"""Catálogo de materiales/perfiles y verificación por factor de seguridad."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from core.builders import BeamDistLoad, BeamSupport, build_beam, self_weight_q
from core.materials import (
    CHANNEL_TORSION_WARNING,
    FAMILY_NOTES,
    MATERIALS,
    SECTIONS,
    THIN_WALL_WARNING,
    chs,
    cold_formed_channel,
    custom,
    family_warnings,
    find_lightest,
    get_section,
    rhs,
    solid_rect,
)
from core.model import SupportType as ST
from core.solver import solve
from core.verification import Status, check_safety, required_section_modulus

# Módulos resistentes de tabla [cm³] para controlar la carga de I y h
_W_TABLE = {"IPE 200": 194.0, "IPE 300": 557.0, "IPE 400": 1156.0, "IPN 200": 214.0, "IPN 300": 653.0}


@pytest.mark.parametrize("name,w_cm3", _W_TABLE.items())
def test_rolled_section_W_matches_table(name, w_cm3):
    assert get_section(name).W == pytest.approx(w_cm3 * 1e3, rel=0.01)


def test_catalog_consistency():
    for fam in SECTIONS.values():
        for s in fam.values():
            assert s.A > 0 and s.I > 0 and s.W > 0
            # Radio de giro físicamente razonable: r < h/2
            assert math.sqrt(s.I / s.A) < s.h / 2


def test_hollow_sections_geometry():
    p = chs("4in", 114.3, 6.02)  # 4" Sch40: A ≈ 20.5 cm², I ≈ 301 cm⁴
    assert p.A == pytest.approx(2048, rel=0.01)
    assert p.I == pytest.approx(3.01e6, rel=0.01)
    t = rhs("100x100x5", 100, 100, 5)
    assert t.A == pytest.approx(100**2 - 90**2)
    assert t.I == pytest.approx((100**4 - 90**4) / 12)


def test_materials_basic():
    a36 = MATERIALS["ASTM A36"]
    assert a36.Sy == 250 and a36.E == 200_000
    assert a36.G == pytest.approx(76_923, rel=1e-3)
    assert all(m.Sy < m.Su for m in MATERIALS.values())


def test_self_weight_ipe200():
    s = get_section("IPE 200")  # 22.4 kg/m
    q = self_weight_q(s, MATERIALS["ASTM A36"])
    assert q == pytest.approx(-22.4 * 9.80665 / 1000, rel=0.01)  # N/mm


def test_find_lightest():
    s = find_lightest(min_W=300e3, family="IPE")
    assert s is not None and s.name == "IPE 240"
    assert find_lightest(min_W=1e12) is None


def _ss_beam(section_name: str, q: float, L: float = 6000.0):
    sec, mat = get_section(section_name), MATERIALS["ASTM A36"]
    m = build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], sec, mat,
                   dist_loads=[BeamDistLoad(0, L, q, q)])
    return m, solve(m)


def test_safety_factor_value():
    m, r = _ss_beam("IPE 200", -10.0)
    chk = check_safety(m, r)
    sigma = 10.0 * 6000**2 / 8 / get_section("IPE 200").W
    assert chk.sigma_max == pytest.approx(sigma, rel=0.01)
    assert chk.fs == pytest.approx(250.0 / chk.sigma_max)
    assert chk.x == pytest.approx(3000.0)
    assert chk.status is Status.ALERT  # σ ≈ 232 MPa -> FS ≈ 1.08


@pytest.mark.parametrize("q,status", [(-3.0, Status.OK), (-10.0, Status.ALERT), (-20.0, Status.FAIL)])
def test_safety_status(q, status):
    m, r = _ss_beam("IPE 200", q)
    assert check_safety(m, r).status is status


def test_required_W_roundtrip():
    W = required_section_modulus(45e6, 250.0, 1.5)
    assert W == pytest.approx(270e3)


# --------------------------------------------------------------------------- #
# Familias UPN, W y C conformado (rama feat/catalog-upn-w-c)
# --------------------------------------------------------------------------- #

# Tolerancia frente a tabla: < 1 % (redondeo de catálogo).
_TAB_REL = 0.01

# UPN — DIN 1026-1: (A [cm²], Ix [cm⁴], Wx [cm³], masa [kg/m]).
# Wx y masa son columnas independientes de la tabla: controlan h (c = h/2) y A.
_UPN_REF = {
    "UPN 100": (13.5, 206.0, 41.2, 10.6),
    "UPN 200": (32.2, 1910.0, 191.0, 25.3),
    "UPN 300": (58.8, 8030.0, 535.0, 46.2),
}

# W — AISC Shapes Database v15.0 (ASTM A6), US -> SI: A [in²]·645,16 = mm²,
# Ix [in⁴]·416 231 = mm⁴, Sx [in³]·16 387 = mm³; masa = designación métrica [kg/m].
_W_REF = {
    "W 150x18": (3.55 * 645.16, 22.1 * 416_231.4, 7.31 * 16_387.06, 18.0),  # W6x12
    "W 200x22.5": (4.44 * 645.16, 48.0 * 416_231.4, 11.8 * 16_387.06, 22.5),  # W8x15
    "W 310x38.7": (7.65 * 645.16, 204.0 * 416_231.4, 33.4 * 16_387.06, 38.7),  # W12x26
}


@pytest.mark.parametrize("name,ref", _UPN_REF.items())
def test_upn_matches_din_1026_table(name, ref):
    """UPN DIN 1026-1: A, Ix de tabla; Wx = Ix/(h/2) y masa = 7850·A contra columnas de tabla."""
    A_cm2, I_cm4, W_cm3, kg_m = ref
    s = get_section(name)
    assert s.family == "UPN"
    assert s.A == pytest.approx(A_cm2 * 100.0, rel=_TAB_REL)
    assert s.I == pytest.approx(I_cm4 * 1e4, rel=_TAB_REL)
    assert s.W == pytest.approx(W_cm3 * 1e3, rel=_TAB_REL)
    assert s.mass_per_m == pytest.approx(kg_m, rel=_TAB_REL)


@pytest.mark.parametrize("name,ref", _W_REF.items())
def test_w_shape_matches_astm_a6_table(name, ref):
    """Perfil W ASTM A6: A, Ix de tabla; Sx = Ix/(d/2) con d real; masa ≈ designación."""
    A, I, W, kg_m = ref
    s = get_section(name)
    assert s.family == "Perfil W"
    assert s.A == pytest.approx(A, rel=_TAB_REL)
    assert s.I == pytest.approx(I, rel=_TAB_REL)
    assert s.W == pytest.approx(W, rel=_TAB_REL)
    assert s.mass_per_m == pytest.approx(kg_m, rel=_TAB_REL)


def _c_linear_method(H: float, B: float, t: float, r_i: float) -> tuple[float, float]:
    """Canal sin labios por el método lineal AISI (Yu & LaBoube, Cold-Formed Steel Design).

    Línea media: r = r_i + t/2; tramo recto de alma a' = H − 2(r_i + t); de ala b' = B − (r_i + t);
    arco de 90°: u = 1,571·r, centroide a 0,637·r del centro de curvatura, I propia = 0,149·r³.
    A = t·(a' + 2b' + 2u);  Ix = t·[a'³/12 + 2b'·(H/2 − t/2)² + 2(u·(H/2 − r_i − t + 0,637r)² + 0,149r³)].
    """
    r = r_i + t / 2.0
    a = H - 2.0 * (r_i + t)
    b = B - (r_i + t)
    u = math.pi / 2.0 * r
    y_arc = H / 2.0 - r_i - t + 0.637 * r
    A = t * (a + 2.0 * b + 2.0 * u)
    I = t * (a**3 / 12.0 + 2.0 * b * (H / 2.0 - t / 2.0) ** 2 + 2.0 * (u * y_arc**2 + 0.149 * r**3))
    return A, I


@pytest.mark.parametrize("name,H,B,t", [
    ("C 80x40x2", 80.0, 40.0, 2.0),
    ("C 140x60x2.5", 140.0, 60.0, 2.5),
    ("C 200x75x3", 200.0, 75.0, 3.0),
])
def test_cold_formed_c_matches_aisi_linear_method(name, H, B, t):
    """Perfil C conformado (r_i = t): A, Ix y Wx = Ix/(H/2) contra el método lineal AISI (< 1 %)."""
    A_ref, I_ref = _c_linear_method(H, B, t, r_i=t)
    s = get_section(name)
    assert s.family == "Perfil C (Conformado)"
    assert s.A == pytest.approx(A_ref, rel=_TAB_REL)
    assert s.I == pytest.approx(I_ref, rel=_TAB_REL)
    assert s.W == pytest.approx(I_ref / (H / 2.0), rel=_TAB_REL)


def test_cold_formed_c_80x40x2_hand_values():
    """C 80x40x2, r_i = 2: A = 2·(72 + 2·36) + 2·π/4·(4² − 2²) = 306,85 mm² (cálculo manual)."""
    s = get_section("C 80x40x2")
    assert s.A == pytest.approx(288.0 + math.pi / 2.0 * 12.0, rel=1e-9)
    assert s.mass_per_m == pytest.approx(2.41, rel=_TAB_REL)


def test_cold_formed_c_sharp_corner_limit():
    """r_i → 0 tiende al canal de esquinas vivas: A = t·(H + 2B − 2t) − 2·t²·(1 − π/4)."""
    H, B, t = 100.0, 50.0, 2.0
    s = cold_formed_channel("C test", H, B, t, r_i=0.0)
    assert s.A == pytest.approx(t * (H + 2 * B - 2 * t) - 2 * t**2 * (1 - math.pi / 4), rel=1e-9)
    with pytest.raises(ValueError):
        cold_formed_channel("C mala", 10.0, 50.0, 4.0)


_NEW_FAMILIES = {
    "UPN": [f"UPN {n}" for n in (80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300)],
    "Perfil W": ["W 150x13", "W 150x18", "W 200x15", "W 200x22.5", "W 250x28.4", "W 310x38.7"],
    "Perfil C (Conformado)": ["C 80x40x2", "C 100x50x2", "C 120x50x2", "C 140x60x2.5",
                              "C 160x60x2.5", "C 200x75x3"],
}


@pytest.mark.parametrize("family,names", _NEW_FAMILIES.items())
def test_new_families_loaded_and_physical(family, names):
    """Todas las secciones cargadas: masa/m > 0, I > 0, simétricas en eje fuerte y en mm."""
    assert list(SECTIONS[family]) == names
    for s in SECTIONS[family].values():
        assert s.mass_per_m > 0 and s.I > 0 and s.A > 0
        assert s.c_top == pytest.approx(s.h / 2) and s.c_bot == pytest.approx(s.h / 2)
        assert 50.0 < s.h < 400.0  # mm (detecta un dato cargado en cm o m)


@pytest.mark.parametrize("family", _NEW_FAMILIES)
def test_new_families_monotonic_with_size(family):
    """Dentro de cada familia, a mayor altura no disminuye I (control de transcripción)."""
    secs = sorted(SECTIONS[family].values(), key=lambda s: (s.h, s.A))
    assert all(a.I <= b.I for a, b in pairwise(secs))


def test_find_lightest_and_suggest_section_new_families():
    """find_lightest / suggest_section operan sobre las familias nuevas.

    Viga simple L = 6 m, q = 10 N/mm: M = qL²/8 = 45e6 N·mm; con FS 1,5 y Sy = 250 MPa,
    W_req = 270e3 mm³ -> el UPN más liviano con W >= W_req es UPN 240 (W = 300 cm³).
    Con flecha L/300 = 20 mm: δ = 5qL⁴/384EI ⇒ I ≥ 4219 cm⁴ -> UPN 260 (UPN 240: δ = 23,4 mm).
    """
    from core.design import suggest_section

    s = find_lightest(min_W=270e3, family="UPN")
    assert s is not None and s.name == "UPN 240"

    L, mat = 6000.0, MATERIALS["ASTM A36"]

    def build(sec):
        return build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], sec, mat,
                          dist_loads=[BeamDistLoad(0, L, -10.0, -10.0)])

    for fam in _NEW_FAMILIES:
        sug = suggest_section(build, fam, fs_min=1.5)
        if sug is not None:
            assert sug.section.family == fam and sug.check.fs >= 1.5
    sug_upn = suggest_section(build, "UPN", fs_min=1.5, deflection_limit_ratio=0.0)
    assert sug_upn is not None and sug_upn.section.name == "UPN 240"
    sug_upn = suggest_section(build, "UPN", fs_min=1.5)  # L/300 por defecto
    assert sug_upn is not None and sug_upn.section.name == "UPN 260"
    assert sug_upn.check.delta_max == pytest.approx(5 * 10.0 * L**4 / (384 * mat.E * 4820e4), rel=1e-6)


def test_every_family_has_a_note():
    """Toda familia que puede elegir la UI (catálogo + constructores) tiene nota de fuente/supuestos."""
    families = set(SECTIONS) | {custom("x", 1.0, 1.0, 1.0).family, solid_rect("x", 1.0, 1.0).family}
    assert families <= set(FAMILY_NOTES)
    assert all(note.strip() for note in FAMILY_NOTES.values())


@pytest.mark.parametrize("family,expected", [
    ("IPE", ()),
    ("Perfil W", ()),
    ("UPN", (CHANNEL_TORSION_WARNING,)),
    ("Perfil C (Conformado)", (CHANNEL_TORSION_WARNING, THIN_WALL_WARNING)),
    ("Familia inexistente", ()),
])
def test_family_warnings(family, expected):
    """Canales (centro de corte fuera del alma) y chapa delgada llevan advertencia de alcance."""
    assert family_warnings(family) == expected
