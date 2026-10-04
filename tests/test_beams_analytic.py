"""Validación del solver contra soluciones analíticas clásicas (Euler-Bernoulli).

Unidades: mm, N, MPa. Las tolerancias son ajustadas porque el solver es exacto
(no hay error de discretización): cualquier desvío indica un bug.
"""

from __future__ import annotations

import math

import pytest

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam
from core.materials import Material, custom
from core.model import SupportType as ST
from core.solver import solve

REL = 1e-6

E = 200_000.0  # MPa
I = 2.0e7  # mm⁴  # noqa: E741
A = 3000.0  # mm²
C = 100.0  # mm
L = 6000.0  # mm
q = -10.0  # N/mm (= 10 kN/m hacia abajo)
P = -20_000.0  # N (20 kN hacia abajo)

MAT = Material("test", E=E, Sy=250.0, Su=400.0)
SEC = custom("test", A=A, I=I, c=C)


def beam(supports, point_loads=None, dist_loads=None, length=L):
    return build_beam(length, supports, SEC, MAT, point_loads, dist_loads)


def uy_at(res, model, x):
    node = min(range(len(model.nodes)), key=lambda k: abs(model.nodes[k][0] - x))
    assert model.nodes[node][0] == pytest.approx(x)
    return res.displacements[node, 1]


# --------------------------------------------------------------------------- #


def test_simply_supported_uniform():
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)],
             dist_loads=[BeamDistLoad(0, L, q, q)], point_loads=[BeamPointLoad(L / 2)])
    r = solve(m)
    w = -q
    assert r.reactions[0][1] == pytest.approx(w * L / 2, rel=REL)
    assert r.extreme("M").value == pytest.approx(w * L**2 / 8, rel=REL)
    assert abs(r.extreme("V").value) == pytest.approx(w * L / 2, rel=REL)
    assert uy_at(r, m, L / 2) == pytest.approx(-5 * w * L**4 / (384 * E * I), rel=REL)
    # Elástica interna exacta: máximo en el centro
    assert r.extreme("uy").value == pytest.approx(-5 * w * L**4 / (384 * E * I), rel=REL)
    # σ = M·c/I
    assert r.extreme("sigma").value == pytest.approx(w * L**2 / 8 * C / I, rel=REL)


def test_simply_supported_single_member_interior_deflection():
    """Sin nodo intermedio: la elástica dentro de la barra debe seguir siendo exacta."""
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)],
             dist_loads=[BeamDistLoad(0, L, q, q)])
    assert len(m.members) == 1
    r = solve(m)
    assert r.extreme("uy").value == pytest.approx(5 * q * L**4 / (384 * E * I), rel=REL)
    assert r.extreme("M").x == pytest.approx(L / 2, rel=REL)


def test_simply_supported_center_point_load():
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)],
             point_loads=[BeamPointLoad(L / 2, Fy=P)])
    r = solve(m)
    assert r.extreme("M").value == pytest.approx(-P * L / 4, rel=REL)
    assert uy_at(r, m, L / 2) == pytest.approx(P * L**3 / (48 * E * I), rel=REL)


def test_simply_supported_eccentric_point_load():
    a, b = 2000.0, 4000.0
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)],
             point_loads=[BeamPointLoad(a, Fy=P)])
    r = solve(m)
    assert r.reactions[0][1] == pytest.approx(-P * b / L, rel=REL)
    assert r.extreme("M").value == pytest.approx(-P * a * b / L, rel=REL)
    assert uy_at(r, m, a) == pytest.approx(P * a**2 * b**2 / (3 * E * I * L), rel=REL)


def test_cantilever_tip_load():
    m = beam([BeamSupport(0, ST.FIXED)], point_loads=[BeamPointLoad(L, Fy=P)])
    r = solve(m)
    Rx, Ry, Mz = r.reactions[0]
    assert Ry == pytest.approx(-P, rel=REL)
    assert Mz == pytest.approx(-P * L, rel=REL)  # antihorario para carga hacia abajo a la derecha
    assert r.extreme("M").value == pytest.approx(P * L, rel=REL)  # hogging (negativo)
    assert uy_at(r, m, L) == pytest.approx(P * L**3 / (3 * E * I), rel=REL)
    assert r.displacements[-1, 2] == pytest.approx(P * L**2 / (2 * E * I), rel=REL)


def test_cantilever_uniform():
    m = beam([BeamSupport(0, ST.FIXED)], dist_loads=[BeamDistLoad(0, L, q, q)])
    r = solve(m)
    assert r.extreme("M").value == pytest.approx(q * L**2 / 2, rel=REL)
    assert uy_at(r, m, L) == pytest.approx(q * L**4 / (8 * E * I), rel=REL)


def test_cantilever_triangular_load_max_at_root():
    """q = 0 en el extremo libre, q0 en el empotramiento: δ = q0 L⁴ / 30EI."""
    m = beam([BeamSupport(L, ST.FIXED)], dist_loads=[BeamDistLoad(0, L, 0.0, q)])
    r = solve(m)
    assert r.extreme("M").value == pytest.approx(q * L**2 / 6, rel=REL)
    assert uy_at(r, m, 0.0) == pytest.approx(q * L**4 / (30 * E * I), rel=REL)


def test_fixed_fixed_uniform():
    m = beam([BeamSupport(0, ST.FIXED), BeamSupport(L, ST.FIXED)],
             dist_loads=[BeamDistLoad(0, L, q, q)])
    r = solve(m)
    w = -q
    assert r.reactions[0][2] == pytest.approx(w * L**2 / 12, rel=REL)
    mr = r.members[0]
    assert mr.M[0] == pytest.approx(-w * L**2 / 12, rel=REL)
    mid = r.extreme("uy")
    assert mid.x == pytest.approx(L / 2, rel=1e-9)
    assert mid.value == pytest.approx(-w * L**4 / (384 * E * I), rel=REL)
    assert max(mr.M) == pytest.approx(w * L**2 / 24, rel=REL)


def test_propped_cantilever_uniform():
    m = beam([BeamSupport(0, ST.FIXED), BeamSupport(L, ST.ROLLER)],
             dist_loads=[BeamDistLoad(0, L, q, q)])
    r = solve(m)
    w = -q
    node_end = len(m.nodes) - 1
    assert r.reactions[node_end][1] == pytest.approx(3 * w * L / 8, rel=REL)
    assert r.reactions[0][2] == pytest.approx(w * L**2 / 8, rel=REL)
    assert max(r.members[0].M) == pytest.approx(9 * w * L**2 / 128, rel=REL)
    xm = (15 - math.sqrt(33)) / 16 * L  # posición de la flecha máxima
    d_exact = -w * xm**2 * (3 * L**2 - 5 * L * xm + 2 * xm**2) / (48 * E * I)
    assert r.extreme("uy").value == pytest.approx(d_exact, rel=REL)


def test_two_span_continuous_uniform():
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER), BeamSupport(2 * L, ST.ROLLER)],
             dist_loads=[BeamDistLoad(0, 2 * L, q, q)], length=2 * L)
    r = solve(m)
    w = -q
    mid = next(n for n in r.reactions if m.nodes[n][0] == pytest.approx(L))
    assert r.reactions[mid][1] == pytest.approx(10 * w * L / 8, rel=REL)
    assert r.extreme("M").value == pytest.approx(-w * L**2 / 8, rel=REL)


def test_partial_and_trapezoidal_load_equilibrium():
    m = beam([BeamSupport(500, ST.PINNED), BeamSupport(5000, ST.ROLLER)],
             dist_loads=[BeamDistLoad(1000, 4000, -5.0, -15.0)],
             point_loads=[BeamPointLoad(6000, Fy=-3000.0, Mz=1e6)])
    r = solve(m)
    total = -(5.0 + 15.0) / 2 * 3000 - 3000.0
    assert sum(v[1] for v in r.reactions.values()) == pytest.approx(-total, rel=REL)
    # Momento antihorario aplicado en el extremo libre -> M interno = +Mz (sagging)
    assert r.members[-1].M[-1] == pytest.approx(1e6, rel=REL)
    # Equilibrio de momentos respecto al origen (cargas + reacciones)
    m_loads = -(5.0 * 3000 * 2500 + 0.5 * 10.0 * 3000 * 3000) - 3000.0 * 6000 + 1e6
    m_reac = sum(model_x * v[1] + v[2] for model_x, v in
                 ((m.nodes[n][0], v) for n, v in r.reactions.items()))
    assert m_loads + m_reac == pytest.approx(0.0, abs=1e-3 * abs(m_loads))


def test_free_end_moment_is_zero():
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(4000, ST.ROLLER)],
             dist_loads=[BeamDistLoad(0, L, q, q)])
    r = solve(m)
    assert r.members[-1].M[-1] == pytest.approx(0.0, abs=1e-3)
    assert r.members[-1].V[-1] == pytest.approx(0.0, abs=1e-6)
    # Momento sobre el apoyo = q·a²/2 (voladizo de 2000 mm)
    assert r.members[0].M[-1] == pytest.approx(q * 2000.0**2 / 2, rel=REL)


def test_rotation_continuity_matches_fe():
    """La pendiente de la elástica interna en x=0 coincide con el giro nodal del FE."""
    m = beam([BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)],
             dist_loads=[BeamDistLoad(0, L, q, q)])
    r = solve(m)
    mr = r.members[0]
    slope0 = (mr.v_local[1] - mr.v_local[0]) / (mr.x[1] - mr.x[0])
    theta = q * L**3 / (24 * E * I)
    assert r.displacements[0, 2] == pytest.approx(theta, rel=REL)
    assert slope0 == pytest.approx(theta, rel=2e-3)
    assert math.isfinite(slope0)
