"""Catálogo de materiales/perfiles y verificación por factor de seguridad."""

from __future__ import annotations

import math

import pytest

from core.builders import BeamDistLoad, BeamSupport, build_beam, self_weight_q
from core.materials import MATERIALS, SECTIONS, chs, find_lightest, get_section, rhs
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
