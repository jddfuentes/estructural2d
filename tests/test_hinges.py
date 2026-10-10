"""Rótulas internas (liberación de Mz en extremos de barra): casos analíticos.

Formulación verificada: condensación estática local del giro liberado
(k_c = k_rr − k_rc·k_cc⁻¹·k_cr ; f_c = f_r − k_rc·k_cc⁻¹·f_c). Referencias: Gere & Weaver,
"Analysis of Framed Structures", cap. 4 (miembros con extremos articulados); McGuire, Gallagher &
Ziemian, "Matrix Structural Analysis", §5.3 (condensación estática).

Unidades: mm, N, MPa, N·mm. El solver es exacto por barra: tolerancia REL = 1e-6 salvo que se
indique otra cosa en el test.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.materials import Material, custom
from core.model import DistributedLoad, LoadDirection, Member, Model, NodalLoad, Support
from core.model import SupportType as ST
from core.solver import (
    Results,
    StructuralError,
    equivalent_nodal_loads,
    local_stiffness,
    released_dofs,
    solve,
    static_condensation,
)

REL = 1e-6
TOL_M = 1.0  # N·mm (= 1e-6 kN·m): criterio de "M = 0" en un extremo NO liberado (redondeo)

E = 200_000.0  # MPa
MAT = Material("test", E=E, Sy=250.0, Su=400.0)
SEC = custom("viga", A=3000.0, I=2.0e7, c=100.0)
COL = custom("col", A=5000.0, I=5.0e7, c=150.0)
q = -10.0  # N/mm (= 10 kN/m hacia abajo)


def _equilibrium(model: Model, r: Results) -> None:
    """ΣFx = ΣFy = ΣMz(origen) = 0 con cargas nodales, distribuidas uniformes GLOBAL_Y y reacciones."""
    fx = fy = mz = 0.0
    for p in model.nodal_loads:
        x, y = model.nodes[p.node]
        fx, fy, mz = fx + p.Fx, fy + p.Fy, mz + p.Mz + x * p.Fy - y * p.Fx
    for d in model.distributed_loads:
        assert d.direction is LoadDirection.GLOBAL_Y and d.q_start == d.q_end
        L = model.member_length(d.member)
        mem = model.members[d.member]
        (x1, _), (x2, _) = model.nodes[mem.i], model.nodes[mem.j]
        W = d.q_start * L
        fy, mz = fy + W, mz + 0.5 * (x1 + x2) * W
    for n, (rx, ry, rm) in r.reactions.items():
        x, y = model.nodes[n]
        fx, fy, mz = fx + rx, fy + ry, mz + rm + x * ry - y * rx
    scale_f = sum(abs(v[1]) + abs(v[0]) for v in r.reactions.values())
    scale_m = scale_f * max(max(abs(x), abs(y)) for x, y in model.nodes)
    assert abs(fx) <= 1e-9 * scale_f
    assert abs(fy) <= 1e-9 * scale_f
    assert abs(mz) <= 1e-9 * scale_m


# --------------------------------------------------------------------------- #
# Elemento: matriz de rigidez y cargas equivalentes condensadas
# --------------------------------------------------------------------------- #

L0, I0, A0 = 4000.0, 2.0e7, 3000.0
EI = E * I0
A3, B3, C3 = 3 * EI / L0**3, 3 * EI / L0**2, 3 * EI / L0


def test_member_defaults_are_rigid():
    """Retrocompatibilidad: Member(i, j, sec, mat) sigue siendo una unión rígida continua."""
    m = Member(0, 1, SEC, MAT)
    assert (m.release_start, m.release_end) == (False, False)
    assert released_dofs(False, False) == ()
    k_old = local_stiffness(E, A0, I0, L0)
    assert np.array_equal(k_old, local_stiffness(E, A0, I0, L0, False, False))


def test_local_stiffness_release_start():
    """Rótula en i: k_flex con 3EI/L³, 3EI/L², 3EI/L; fila/columna de θ_i nulas."""
    k = local_stiffness(E, A0, I0, L0, release_start=True)
    v_i, t_i, v_j, t_j = 1, 2, 4, 5
    expected = np.array([[A3, 0, -A3, B3], [0, 0, 0, 0], [-A3, 0, A3, -B3], [B3, 0, -B3, C3]])
    idx = [v_i, t_i, v_j, t_j]
    assert k[np.ix_(idx, idx)] == pytest.approx(expected, rel=1e-12)
    assert np.all(k[t_i, :] == 0.0) and np.all(k[:, t_i] == 0.0)
    assert k[0, 0] == pytest.approx(E * A0 / L0) and k[0, 3] == pytest.approx(-E * A0 / L0)


def test_local_stiffness_release_end():
    """Rótula en j: formulación simétrica, fila/columna de θ_j nulas."""
    k = local_stiffness(E, A0, I0, L0, release_end=True)
    idx = [1, 2, 4, 5]
    expected = np.array([[A3, B3, -A3, 0], [B3, C3, -B3, 0], [-A3, -B3, A3, 0], [0, 0, 0, 0]])
    assert k[np.ix_(idx, idx)] == pytest.approx(expected, rel=1e-12)
    assert np.all(k[5, :] == 0.0) and np.all(k[:, 5] == 0.0)


def test_local_stiffness_both_releases_is_axial_only():
    """Biela: rigidez a flexión y corte nula; sólo EA/L."""
    k = local_stiffness(E, A0, I0, L0, release_start=True, release_end=True)
    ea = E * A0 / L0
    expected = np.zeros((6, 6))
    expected[np.ix_([0, 3], [0, 3])] = [[ea, -ea], [-ea, ea]]
    assert np.array_equal(k, expected)


@pytest.mark.parametrize("rs, re", [(True, False), (False, True), (True, True)])
def test_closed_form_matches_static_condensation(rs, re):
    """La forma cerrada coincide con la condensación estática numérica de la matriz completa."""
    k_full = local_stiffness(E, A0, I0, L0)
    kc, fc = static_condensation(k_full, np.zeros(6), released_dofs(rs, re))
    assert kc == pytest.approx(local_stiffness(E, A0, I0, L0, rs, re), abs=1e-9 * EI / L0)
    assert np.all(fc == 0.0)


@pytest.mark.parametrize(
    "rs, re, expected",
    [
        # (Vi, mi, Vj, mj) de empotramiento con carga uniforme q (f_eq: + según y local / antihorario)
        (False, False, (1 / 2, 1 / 12, 1 / 2, -1 / 12)),
        (True, False, (3 / 8, 0.0, 5 / 8, -1 / 8)),  # Vi = 3qL/8, Vj = 5qL/8, mj = −qL²/8
        (False, True, (5 / 8, 1 / 8, 3 / 8, 0.0)),  # Vi = 5qL/8, Vj = 3qL/8, mi = +qL²/8
        (True, True, (1 / 2, 0.0, 1 / 2, 0.0)),  # biela: Vi = Vj = qL/2
    ],
)
def test_fixed_end_forces_uniform_load(rs, re, expected):
    f = equivalent_nodal_loads((0.0, 0.0), (q, q), L0)
    f = static_condensation(local_stiffness(E, A0, I0, L0), f, released_dofs(rs, re))[1]
    vi, mi, vj, mj = expected
    assert f[[1, 4]] == pytest.approx([vi * q * L0, vj * q * L0], rel=1e-12)
    assert f[[2, 5]] == pytest.approx([mi * q * L0**2, mj * q * L0**2], rel=1e-12, abs=1e-6)
    released = [d for d, flag in ((2, rs), (5, re)) if flag]
    assert np.all(f[released] == 0.0)


# --------------------------------------------------------------------------- #
# Test 1 — Viga Gerber: dos tramos de 4 m, tres apoyos, rótula interna
# --------------------------------------------------------------------------- #
#
#   A (articulado) x=0 ──── B (móvil) x=4 m ── rótula x=5 m ──── C (móvil) x=8 m, carga q uniforme.
#   Isostática. Tramo rótula–C (a = 3 m) simplemente apoyado: V_rótula = R_C = |q|·a/2.
#   Voladizo A–B–rótula: R_B·4 = |q|·5·2.5 + |q|a/2·5  ->  R_B = 5|q|·m, R_A = 1.5|q|·m.
#   M_B = −(|q|·1²/2 + |q|a/2·1) = −2|q|·m² ;  M_máx(A–B) = R_A²/(2|q|) ; M_máx(rótula–C) = |q|a²/8.

XA, XB, XH, XC = 0.0, 4000.0, 5000.0, 8000.0
a = XC - XH
w = -q  # intensidad (positiva) [N/mm]


def _gerber(where: str) -> Model:
    left_end = where in ("left", "both")
    right_start = where in ("right", "both")
    return Model(
        nodes=[(XA, 0.0), (XB, 0.0), (XH, 0.0), (XC, 0.0)],
        members=[
            Member(0, 1, SEC, MAT),
            Member(1, 2, SEC, MAT, release_end=left_end),
            Member(2, 3, SEC, MAT, release_start=right_start),
        ],
        supports=[Support(0, ST.PINNED), Support(1, ST.ROLLER), Support(3, ST.ROLLER)],
        distributed_loads=[DistributedLoad(k, q, q) for k in range(3)],
    )


@pytest.mark.parametrize("where", ["left", "right", "both"])
def test_gerber_beam_moment_zero_at_hinge(where):
    """Viga Gerber: M(rótula) = 0, reacciones y momentos isostáticos exactos, equilibrio global."""
    m = _gerber(where)
    r = solve(m)
    _equilibrium(m, r)

    R_C = w * a / 2
    R_B = (w * XH * XH / 2 + R_C * XH) / XB
    R_A = w * XC - R_B - R_C
    assert r.reactions[0][1] == pytest.approx(R_A, rel=REL)
    assert r.reactions[1][1] == pytest.approx(R_B, rel=REL)
    assert r.reactions[3][1] == pytest.approx(R_C, rel=REL)
    assert (R_A, R_B, R_C) == pytest.approx((15_000.0, 50_000.0, 15_000.0), rel=1e-12)

    # Momento en la coordenada exacta de la rótula (ambos lados): 0 con error < 1e-6 kN·m
    left, right = r.members[1], r.members[2]
    assert left.x[-1] == XH - XB and right.x[0] == 0.0
    assert abs(left.M[-1]) < TOL_M and abs(right.M[0]) < TOL_M
    if where in ("left", "both"):
        assert left.M[-1] == 0.0 and left.end_forces[5] == 0.0
    if where in ("right", "both"):
        assert right.M[0] == 0.0 and right.end_forces[2] == 0.0

    # Momentos característicos
    M_B = -(w * (XH - XB) ** 2 / 2 + R_C * (XH - XB))
    assert r.members[0].M[-1] == pytest.approx(M_B, rel=REL)
    assert M_B == pytest.approx(-20.0e6, rel=1e-12)  # −20 kN·m
    assert float(np.max(r.members[0].M)) == pytest.approx(R_A**2 / (2 * w), rel=REL)
    assert float(np.max(right.M)) == pytest.approx(w * a**2 / 8, rel=REL)
    # Flecha en la rótula = punta del voladizo B–rótula (b = 1 m) con P = R_C y q:
    #   v = θ_B·b − q·b⁴/8EI − R_C·b³/3EI ,  θ_B = q·l³/24EI + M_B·l/3EI  (l = 4 m; aquí θ_B = 0)
    EI_ = E * SEC.I
    b, l_ = XH - XB, XB - XA
    theta_B = w * l_**3 / (24 * EI_) + M_B * l_ / (3 * EI_)
    v_hinge = theta_B * b - w * b**4 / (8 * EI_) - R_C * b**3 / (3 * EI_)
    assert r.displacements[2, 1] == pytest.approx(v_hinge, rel=REL)
    assert v_hinge == pytest.approx(-1.5625, rel=1e-12)
    # La flecha es continua en la rótula (mismo nodo) aunque el giro no lo sea
    assert left.displacement[-1, 1] == pytest.approx(right.displacement[0, 1], rel=REL)
    assert left.v_local[-1] == pytest.approx(v_hinge, rel=REL)


def test_gerber_hinge_without_extra_support_is_mechanism():
    """Viga simplemente apoyada + rótula interna = mecanismo -> StructuralError."""
    m = Model(
        nodes=[(0.0, 0.0), (3000.0, 0.0), (6000.0, 0.0)],
        members=[Member(0, 1, SEC, MAT, release_end=True), Member(1, 2, SEC, MAT)],
        supports=[Support(0, ST.PINNED), Support(2, ST.ROLLER)],
        distributed_loads=[DistributedLoad(0, q, q), DistributedLoad(1, q, q)],
    )
    with pytest.raises(StructuralError):
        solve(m)


# --------------------------------------------------------------------------- #
# Test 2 — Barra biarticulada aislada con carga distribuida
# --------------------------------------------------------------------------- #

L2 = 6000.0


def _isolated(rs: bool, re: bool) -> Model:
    return Model(
        nodes=[(0.0, 0.0), (L2, 0.0)],
        members=[Member(0, 1, SEC, MAT, release_start=rs, release_end=re)],
        supports=[Support(0, ST.PINNED), Support(1, ST.ROLLER)],
        distributed_loads=[DistributedLoad(0, q, q)],
    )


def test_pin_ended_member_uniform_load():
    """Biela con q: M_máx = qL²/8 al centro, M = 0 exacto en extremos, δ = 5qL⁴/384EI."""
    m = _isolated(True, True)
    r = solve(m)
    _equilibrium(m, r)
    mr = r.members[0]
    assert mr.M[0] == 0.0 and mr.M[-1] == 0.0
    assert mr.sigma_top[0] == 0.0 and mr.sigma_bot[-1] == 0.0  # sin axil: σ = 0 en las rótulas
    ext = r.extreme("M")
    assert ext.value == pytest.approx(w * L2**2 / 8, rel=REL)
    assert ext.x == pytest.approx(L2 / 2, rel=REL)
    assert r.extreme("uy").value == pytest.approx(-5 * w * L2**4 / (384 * E * SEC.I), rel=REL)
    assert mr.V[0] == pytest.approx(w * L2 / 2, rel=REL) and mr.V[-1] == pytest.approx(-w * L2 / 2, rel=REL)
    # Nudos sin rigidez a giro (todas las barras articuladas): giro nodal informado = 0
    assert r.displacements[0, 2] == 0.0 and r.displacements[1, 2] == 0.0


@pytest.mark.parametrize("rs, re", [(True, False), (False, True), (True, True)])
def test_releases_on_simply_supported_beam_change_nothing(rs, re):
    """Liberar M donde ya es 0 (apoyos simples) no altera N-V-M ni la elástica.

    Comparación estación por estación con tolerancia REL referida al PICO de cada diagrama
    (abs = REL·máx|·|), no al valor local: V y v pasan por cero (centro y apoyos), donde una
    tolerancia relativa pura exige igualdad bit a bit del redondeo (~1e-11 N sobre 3e4 N, que
    cambia entre versiones de numpy/BLAS y entre estaciones x insertadas en V = 0).
    """
    ref = solve(_isolated(False, False))
    r = solve(_isolated(rs, re))
    a_, b_ = ref.members[0], r.members[0]
    assert b_.x == pytest.approx(a_.x, abs=REL * L2)
    assert b_.M == pytest.approx(a_.M, abs=TOL_M)
    assert b_.V == pytest.approx(a_.V, abs=REL * float(np.max(np.abs(a_.V))))
    assert b_.v_local == pytest.approx(a_.v_local, abs=REL * float(np.max(np.abs(a_.v_local))))


# --------------------------------------------------------------------------- #
# Rótula en un solo extremo: viga biempotrada con rótula = empotrada-articulada
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side", ["start", "end"])
def test_single_release_gives_propped_cantilever(side):
    """Empotrada–articulada con q: R_art = 3qL/8, R_emp = 5qL/8, M_emp = −qL²/8, M+ = 9qL²/128."""
    rs = side == "start"
    m = Model(
        nodes=[(0.0, 0.0), (L2, 0.0)],
        members=[Member(0, 1, SEC, MAT, release_start=rs, release_end=not rs)],
        supports=[Support(0, ST.FIXED), Support(1, ST.FIXED)],
        distributed_loads=[DistributedLoad(0, q, q)],
    )
    r = solve(m)
    _equilibrium(m, r)
    hinge, fixed = (0, 1) if rs else (1, 0)
    assert r.reactions[hinge][1] == pytest.approx(3 * w * L2 / 8, rel=REL)
    assert r.reactions[fixed][1] == pytest.approx(5 * w * L2 / 8, rel=REL)
    assert r.reactions[hinge][2] == 0.0
    mr = r.members[0]
    M_hinge, M_fixed = (mr.M[0], mr.M[-1]) if rs else (mr.M[-1], mr.M[0])
    assert M_hinge == 0.0
    assert M_fixed == pytest.approx(-w * L2**2 / 8, rel=REL)
    k = int(np.argmax(mr.M))  # máximo positivo (el extremo |M| es el de empotramiento)
    assert mr.M[k] == pytest.approx(9 * w * L2**2 / 128, rel=REL)
    assert mr.x[k] == pytest.approx(3 * L2 / 8 if rs else 5 * L2 / 8, rel=REL)


# --------------------------------------------------------------------------- #
# Test 3 — Pórtico con dintel articulado a las columnas (biela)
# --------------------------------------------------------------------------- #
#
#   Columnas empotradas en la base (voladizos de altura H) unidas por un dintel biarticulado.
#   El dintel trabaja como viga simplemente apoyada (M_máx = qS²/8) y sólo transmite axil
#   horizontal a las columnas. Con carga lateral P en el nudo izquierdo, el reparto entre
#   voladizos (k_c = 3EI_c/H³) en serie con la biela (k_a = EA_b/S) es exacto:
#       P_der = P / (2 + k_c/k_a) ;  P_izq = P − P_der ;  M_base = P_k·H ;  N_dintel = −P_der.

S, H = 8000.0, 4000.0
P_LAT = 10_000.0  # N
q_beam = -15.0  # N/mm


def _portal(lateral: float) -> Model:
    return Model(
        nodes=[(0.0, 0.0), (0.0, H), (S, H), (S, 0.0)],
        members=[
            Member(0, 1, COL, MAT),
            Member(1, 2, SEC, MAT, release_start=True, release_end=True),
            Member(3, 2, COL, MAT),
        ],
        supports=[Support(0, ST.FIXED), Support(3, ST.FIXED)],
        nodal_loads=[NodalLoad(1, Fx=lateral)] if lateral else [],
        distributed_loads=[DistributedLoad(1, q_beam, q_beam)],
    )


def test_portal_pinned_beam_gravity_no_moment_in_columns():
    """Sólo gravedad: M_col = 0 en toda la columna; el dintel es una viga simple (qS²/8)."""
    m = _portal(0.0)
    r = solve(m)
    _equilibrium(m, r)
    beam = r.members[1]
    assert beam.M[0] == 0.0 and beam.M[-1] == 0.0
    assert r.extreme("M").value == pytest.approx(-q_beam * S**2 / 8, rel=REL)
    for col in (r.members[0], r.members[2]):
        assert float(np.max(np.abs(col.M))) < TOL_M
        assert col.N[0] == pytest.approx(q_beam * S / 2, rel=REL)  # compresión = |q|S/2
    for n in (0, 3):
        assert abs(r.reactions[n][2]) < TOL_M


def test_portal_pinned_beam_lateral_load():
    """Gravedad + lateral: M = 0 en la cabeza de columnas, reparto exacto P_izq/P_der, equilibrio."""
    m = _portal(P_LAT)
    r = solve(m)
    _equilibrium(m, r)
    col_l, beam, col_r = r.members
    assert beam.M[0] == 0.0 and beam.M[-1] == 0.0
    assert abs(col_l.M[-1]) < TOL_M and abs(col_r.M[-1]) < TOL_M  # nudo = articulación pura

    k_c = 3 * E * COL.I / H**3
    k_a = E * SEC.A / S
    P_r = P_LAT / (2 + k_c / k_a)
    P_l = P_LAT - P_r
    assert float(np.mean(beam.N)) == pytest.approx(-P_r, rel=REL)
    assert r.reactions[0][0] == pytest.approx(-P_l, rel=REL)
    assert r.reactions[3][0] == pytest.approx(-P_r, rel=REL)
    assert r.reactions[0][2] == pytest.approx(P_l * H, rel=REL)
    assert r.reactions[3][2] == pytest.approx(P_r * H, rel=REL)
    # La cabeza de columna gira como voladizo libre (θ = −P·H²/2EI, horario): el dintel no la restringe
    theta_col = r.displacements[1, 2]
    assert theta_col == pytest.approx(-P_l * H**2 / (2 * E * COL.I), rel=REL)


def test_moment_on_fully_hinged_node_raises():
    """Momento aplicado en un nudo donde todas las barras están articuladas -> StructuralError."""
    m = _isolated(True, True)
    m.nodal_loads.append(NodalLoad(1, Mz=1.0e6))
    with pytest.raises(StructuralError, match="rótula"):
        solve(m)


def test_solve_does_not_mutate_model_with_hinges():
    m = _gerber("both")
    before = (list(m.nodes), list(m.members), list(m.supports), list(m.distributed_loads))
    solve(m)
    assert (m.nodes, m.members, m.supports, m.distributed_loads) == before
