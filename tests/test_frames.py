"""Pórticos: casos analíticos, equilibrio global, inestabilidad y verificación cruzada con anastruct."""

from __future__ import annotations

import pytest

from core.builders import BeamPointLoad, BeamSupport, build_beam, build_portal_frame
from core.materials import Material, custom
from core.model import DistributedLoad, LoadDirection, Member, Model, NodalLoad, Support
from core.model import SupportType as ST
from core.solver import StructuralError, solve

E = 200_000.0
MAT = Material("test", E=E, Sy=250.0, Su=400.0)
COL = custom("col", A=5000.0, I=5.0e7, c=150.0)
RIGID = custom("rigid", A=1e9, I=1e15, c=150.0)  # viga "infinitamente" rígida
S, H = 8000.0, 4000.0
Hx = 10_000.0  # N


def test_portal_fixed_base_lateral_rigid_beam():
    m = build_portal_frame(S, H, COL, RIGID, MAT, ST.FIXED, ST.FIXED, lateral_load=Hx)
    r = solve(m)
    I = COL.I  # noqa: E741
    sway = r.displacements[1, 0]
    assert sway == pytest.approx(Hx * H**3 / (24 * E * I), rel=2e-3)  # incluye def. axial
    # Momento en la base de cada columna = H·h/4
    for n in (0, len(m.nodes) - 1):
        assert abs(r.reactions[n][2]) == pytest.approx(Hx * H / 4, rel=2e-3)
        assert r.reactions[n][0] == pytest.approx(-Hx / 2, rel=2e-3)


def test_portal_pinned_base_lateral_rigid_beam():
    m = build_portal_frame(S, H, COL, RIGID, MAT, ST.PINNED, ST.PINNED, lateral_load=Hx)
    r = solve(m)
    assert r.displacements[1, 0] == pytest.approx(Hx * H**3 / (6 * E * COL.I), rel=2e-3)
    assert abs(r.members[0].M[-1]) == pytest.approx(Hx * H / 2, rel=2e-3)


def test_portal_global_equilibrium_gravity_and_lateral():
    q = -15.0
    P = BeamPointLoad(3000.0, Fy=-25_000.0)
    m = build_portal_frame(S, H, COL, COL, MAT, ST.PINNED, ST.FIXED, beam_q=q,
                           lateral_load=Hx, beam_point_loads=[P], self_weight=True)
    r = solve(m)
    Fx = Hx + sum(n.Fx for n in m.nodal_loads if n.Fx and n.node != 1)
    Fy = sum(n.Fy for n in m.nodal_loads)
    for d in m.distributed_loads:
        assert d.direction is LoadDirection.GLOBAL_Y
        Fy += 0.5 * (d.q_start + d.q_end) * m.member_length(d.member)
    assert sum(v[0] for v in r.reactions.values()) == pytest.approx(-Hx, rel=1e-9)
    assert sum(v[1] for v in r.reactions.values()) == pytest.approx(-Fy, rel=1e-9)
    assert Fx == pytest.approx(Hx)


def test_inclined_member_global_load_projection():
    """Barra inclinada en voladizo con carga vertical global: M empotramiento = q·L·(Lx/2)."""
    Lx, Ly, q = 3000.0, 4000.0, -2.0  # L = 5000
    m = Model(nodes=[(0, 0), (Lx, Ly)], members=[Member(0, 1, COL, MAT)],
              supports=[Support(0, ST.FIXED)],
              distributed_loads=[DistributedLoad(0, q, q, LoadDirection.GLOBAL_Y)])
    r = solve(m)
    Rx, Ry, Mz = r.reactions[0]
    assert Rx == pytest.approx(0.0, abs=1e-6)
    assert Ry == pytest.approx(-q * 5000.0, rel=1e-9)
    assert Mz == pytest.approx(-q * 5000.0 * Lx / 2, rel=1e-9)


def test_global_x_load_on_column():
    """Columna empotrada con viento uniforme w: M base = w·h²/2, δ = w·h⁴/8EI."""
    w = 1.5
    m = Model(nodes=[(0, 0), (0, H)], members=[Member(0, 1, COL, MAT)],
              supports=[Support(0, ST.FIXED)],
              distributed_loads=[DistributedLoad(0, w, w, LoadDirection.GLOBAL_X)])
    r = solve(m)
    assert r.reactions[0][2] == pytest.approx(w * H**2 / 2, rel=1e-9)  # reacción antihoraria
    assert r.displacements[1, 0] == pytest.approx(w * H**4 / (8 * E * COL.I), rel=1e-9)


@pytest.mark.parametrize(
    "supports",
    [
        [BeamSupport(0, ST.PINNED)],  # gira libremente
        [BeamSupport(0, ST.ROLLER), BeamSupport(5000, ST.ROLLER)],  # sin restricción horizontal
        [BeamSupport(0, ST.ROLLER)],
    ],
)
def test_mechanism_raises(supports):
    m = build_beam(6000.0, supports, COL, MAT, point_loads=[BeamPointLoad(3000.0, Fy=-1000.0)])
    with pytest.raises(StructuralError):
        solve(m)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        solve(Model(nodes=[(0, 0), (0, 0)], members=[Member(0, 1, COL, MAT)],
                    supports=[Support(0, ST.FIXED)]))
    with pytest.raises(ValueError):
        build_beam(6000.0, [BeamSupport(7000.0, ST.FIXED)], COL, MAT)


def test_solve_is_pure():
    m = build_portal_frame(S, H, COL, COL, MAT, beam_q=-10.0, lateral_load=Hx)
    snapshot = (list(m.nodes), list(m.members), list(m.nodal_loads), list(m.distributed_loads))
    r1, r2 = solve(m), solve(m)
    assert snapshot == (m.nodes, m.members, m.nodal_loads, m.distributed_loads)
    assert (r1.displacements == r2.displacements).all()


def test_nodal_moment_only():
    """Viga simplemente apoyada con momento en el extremo: θ_A = M L / 3EI."""
    L, M0 = 6000.0, 5e6
    m = build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], COL, MAT,
                   point_loads=[BeamPointLoad(0.0, Mz=M0)])
    r = solve(m)
    assert r.displacements[0, 2] == pytest.approx(M0 * L / (3 * E * COL.I), rel=1e-9)
    assert r.reactions[0][1] == pytest.approx(M0 / L, rel=1e-9)


# --------------------------------------------------------------------------- #
# Verificación cruzada con anastruct (opcional: requirements-dev.txt)
# --------------------------------------------------------------------------- #


def test_cross_check_anastruct_portal():
    anastruct = pytest.importorskip("anastruct")
    q = -12.0
    m = build_portal_frame(S, H, COL, COL, MAT, ST.FIXED, ST.PINNED, beam_q=q, lateral_load=Hx)
    r = solve(m)

    ss = anastruct.SystemElements()
    EA, EI = E * COL.A, E * COL.I
    ss.add_element(location=[[0, 0], [0, H]], EA=EA, EI=EI)
    ss.add_element(location=[[0, H], [S, H]], EA=EA, EI=EI)
    ss.add_element(location=[[S, H], [S, 0]], EA=EA, EI=EI)
    ss.add_support_fixed(node_id=1)
    ss.add_support_hinged(node_id=4)
    ss.q_load(q=q, element_id=2, direction="y")
    ss.point_load(node_id=2, Fx=Hx)
    ss.solve()

    # Reacciones: comparar magnitudes (anastruct usa su propia convención de signos)
    ra = {n["id"]: n for n in ss.get_node_results_system()}
    ours = r.reactions
    assert abs(ra[1]["Fy"]) == pytest.approx(abs(ours[0][1]), rel=1e-6)
    assert abs(ra[1]["Tz"]) == pytest.approx(abs(ours[0][2]), rel=1e-6)
    assert abs(ra[4]["Fx"]) == pytest.approx(abs(ours[len(m.nodes) - 1][0]), rel=1e-6)
    # Desplazamiento lateral del nudo superior izquierdo
    ux_a = ss.get_node_displacements(node_id=2)["ux"]
    assert abs(ux_a) == pytest.approx(abs(r.displacements[1, 0]), rel=1e-6)
    # Momento máximo en la viga
    m_a = max(abs(v) for v in ss.get_element_result_range("moment"))
    m_o = max(abs(mr.M).max() for mr in r.members)
    assert m_a == pytest.approx(m_o, rel=1e-3)


def test_nodal_load_dataclass_defaults():
    assert NodalLoad(0) == NodalLoad(0, 0.0, 0.0, 0.0)
