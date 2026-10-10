"""Apoyos elásticos (resortes nodales kx, ky, krz): casos analíticos.

Formulación verificada: la rigidez de cada resorte se suma a la diagonal de K global en su GDL
(ux, uy, rz); un GDL con resorte y sin vínculo rígido es libre; la reacción del resorte sobre la
estructura es R = −k·u. Referencias: McGuire, Gallagher & Ziemian, "Matrix Structural Analysis",
§2.6 (apoyos elásticos); Timoshenko & Gere, "Mechanics of Materials" (flechas de voladizo y de
viga simple: PL³/3EI, PL³/48EI, 5qL⁴/384EI); viga continua de dos tramos iguales (reacción central
5qL/8 y momento −qa²/8 con L = 2a).

Unidades: mm, N, MPa, N·mm; kx, ky en N/mm y krz en N·mm/rad. El solver es exacto por barra:
tolerancia REL = 1e-6 salvo que se indique otra cosa en el test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from core.materials import Material, custom
from core.model import DistributedLoad, LoadDirection, Member, Model, NodalLoad, Spring, Support
from core.model import SupportType as ST
from core.solver import Results, StructuralError, solve

REL = 1e-6
RESIDUAL = 1e-12  # residuo relativo admisible de equilibrio global (ΣR + ΣF = 0)

E = 200_000.0  # MPa
MAT = Material("test", E=E, Sy=250.0, Su=400.0)
I0 = 2.0e7  # mm⁴
SEC = custom("viga", A=3000.0, I=I0, c=100.0)
COL = custom("col", A=5000.0, I=5.0e7, c=150.0)
EI = E * I0  # N·mm²
P = 10_000.0  # N
q = -10.0  # N/mm (= 10 kN/m hacia abajo)


def _equilibrium_residual(model: Model, r: Results) -> tuple[float, float, float]:
    """(ΣFx, ΣFy, ΣMz respecto del origen) / escala, con cargas + reacciones (rígidas y de resortes).

    Admite cargas nodales y distribuidas uniformes GLOBAL_Y. La escala es la suma de los módulos de
    las cargas (fuerza) y de sus momentos respecto del origen: residuo adimensional.
    """
    fx = fy = mz = 0.0
    sf = sm = 0.0
    for p in model.nodal_loads:
        x, y = model.nodes[p.node]
        fx, fy, mz = fx + p.Fx, fy + p.Fy, mz + p.Mz + x * p.Fy - y * p.Fx
        sf += abs(p.Fx) + abs(p.Fy)
        sm += abs(p.Mz) + abs(x * p.Fy) + abs(y * p.Fx)
    for d in model.distributed_loads:
        assert d.direction is LoadDirection.GLOBAL_Y and d.q_start == d.q_end
        mem = model.members[d.member]
        (x1, _), (x2, _) = model.nodes[mem.i], model.nodes[mem.j]
        W = d.q_start * model.member_length(d.member)
        xc = 0.5 * (x1 + x2)
        fy, mz = fy + W, mz + xc * W
        sf += abs(W)
        sm += abs(xc * W)
    for n, (rx, ry, rm) in r.reactions.items():
        x, y = model.nodes[n]
        fx, fy, mz = fx + rx, fy + ry, mz + rm + x * ry - y * rx
    sm = max(sm, sf * max(max(abs(x), abs(y)) for x, y in model.nodes))
    return abs(fx) / sf, abs(fy) / sf, abs(mz) / sm


# --------------------------------------------------------------------------- #
# Modelo de datos: validación y retrocompatibilidad
# --------------------------------------------------------------------------- #


def test_spring_defaults_and_model_backward_compatible():
    """Spring(node) = sin rigidez; Model() sin resortes sigue funcionando igual que antes."""
    sp = Spring(3)
    assert (sp.kx, sp.ky, sp.krz) == (0.0, 0.0, 0.0)
    assert sp.stiffness == (0.0, 0.0, 0.0)
    assert Model().springs == []


@pytest.mark.parametrize("field_name", ["kx", "ky", "krz"])
@pytest.mark.parametrize("bad", [-1.0, -1e-12, float("nan"), float("inf")])
def test_spring_rejects_negative_or_non_finite_stiffness(field_name, bad):
    """k < 0 (o NaN / ∞) no tiene sentido físico -> ValueError al construir el resorte."""
    with pytest.raises(ValueError, match=field_name):
        Spring(0, **{field_name: bad})


def _bar(n_nodes: int = 2, length: float = 4000.0) -> Model:
    xs = np.linspace(0.0, length, n_nodes)
    return Model(
        nodes=[(float(x), 0.0) for x in xs],
        members=[Member(k, k + 1, SEC, MAT) for k in range(n_nodes - 1)],
    )


def test_spring_on_missing_node_or_duplicated_raises():
    m = _bar()
    m.supports = [Support(0, ST.FIXED)]
    m.springs = [Spring(5, ky=1.0)]
    with pytest.raises(ValueError, match="inexistente"):
        m.validate()
    m.springs = [Spring(1, kx=1.0), Spring(1, ky=1.0)]
    with pytest.raises(ValueError, match="más de un resorte"):
        m.validate()


def test_model_without_any_support_or_spring_raises():
    with pytest.raises(ValueError, match="no tiene apoyos"):
        _bar().validate()


def test_springs_only_model_is_valid():
    """Una estructura apoyada sólo en resortes es válida (no exige `Support`)."""
    m = _bar()
    m.springs = [Spring(0, kx=1e3, ky=1e3, krz=1e9)]
    m.validate()


def test_zero_stiffness_spring_changes_nothing():
    """Resortes con k = 0 no alteran K: desplazamientos y reacciones idénticos bit a bit."""
    base = _bar(3, 6000.0)
    base.supports = [Support(0, ST.PINNED), Support(2, ST.ROLLER)]
    base.distributed_loads = [DistributedLoad(0, q, q), DistributedLoad(1, q, q)]
    with_springs = Model(base.nodes, base.members, base.supports, [], base.distributed_loads,
                         springs=[Spring(1), Spring(2)])
    r0, r1 = solve(base), solve(with_springs)
    assert np.array_equal(r0.displacements, r1.displacements)
    for n in r0.reactions:
        assert r0.reactions[n] == r1.reactions[n]
    assert r1.reactions[1] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9 * abs(q) * 6000.0)  # redondeo


def test_solve_does_not_mutate_model_with_springs():
    m = _cantilever(2, 2.0 * EI / 4000.0)
    before = (list(m.nodes), list(m.members), list(m.supports), list(m.springs))
    solve(m)
    assert (m.nodes, m.members, m.supports, m.springs) == before


# --------------------------------------------------------------------------- #
# Test 1: voladizo con resorte rotacional en la base
# --------------------------------------------------------------------------- #

LC = 4000.0  # mm


def _cantilever(n_nodes: int, k_theta: float) -> Model:
    """Voladizo horizontal: base articulada + resorte rotacional krz = Kθ; P hacia abajo en la punta."""
    m = _bar(n_nodes, LC)
    m.supports = [Support(0, ST.PINNED)]
    m.springs = [Spring(0, krz=k_theta)]
    m.nodal_loads = [NodalLoad(n_nodes - 1, Fy=-P)]
    return m


@pytest.mark.parametrize("n_nodes", [2, 5])  # barra única vs mallada: mismo resultado
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])  # Kθ = α·EI/L
def test_cantilever_rotational_spring_base(n_nodes, alpha):
    """θ₀ = P·L/Kθ (horario) ; δ = P·L³/(3EI) + θ₀·L ; M_reacción = −Kθ·θ₀ = P·L ; M(0) = −P·L."""
    k_theta = alpha * EI / LC
    r = solve(_cantilever(n_nodes, k_theta))
    theta0 = P * LC / k_theta
    delta = P * LC**3 / (3 * EI) + theta0 * LC
    tip = n_nodes - 1
    assert r.displacements[0, 2] == pytest.approx(-theta0, rel=REL)  # giro horario (−)
    assert r.displacements[tip, 1] == pytest.approx(-delta, rel=REL)  # flecha hacia abajo (−)
    assert r.displacements[tip, 2] == pytest.approx(-(P * LC**2 / (2 * EI) + theta0), rel=REL)
    rx, ry, rm = r.reactions[0]
    assert ry == pytest.approx(P, rel=REL)
    assert rx == pytest.approx(0.0, abs=1e-9 * P)
    assert rm == pytest.approx(-k_theta * r.displacements[0, 2], rel=1e-12)  # R = −k·u
    assert rm == pytest.approx(P * LC, rel=REL)
    assert r.members[0].M[0] == pytest.approx(-P * LC, rel=REL)  # hogging en la base
    assert r.extreme("uy").value == pytest.approx(-delta, rel=REL)


def test_cantilever_stiff_rotational_spring_tends_to_fixed():
    """Kθ → ∞: δ → P·L³/(3EI) (empotramiento perfecto)."""
    r = solve(_cantilever(2, 1e9 * EI / LC))
    assert r.displacements[1, 1] == pytest.approx(-P * LC**3 / (3 * EI), rel=1e-8)


def test_cantilever_without_rotational_stiffness_is_mechanism():
    """Kθ = 0 con base articulada: mecanismo -> StructuralError (nunca resultados basura)."""
    with pytest.raises(StructuralError):
        solve(_cantilever(2, 0.0))


# --------------------------------------------------------------------------- #
# Test 2: viga simplemente apoyada con apoyo vertical elástico central
# --------------------------------------------------------------------------- #

A_SPAN = 3000.0  # mm, semiluz
LT = 2 * A_SPAN  # luz total
K_BEAM = 48 * EI / LT**3  # rigidez de la viga simple en el centro (P = K·δ, δ = PL³/48EI)
DELTA0 = 5 * q * LT**4 / (384 * EI)  # flecha central de la viga simple (sin resorte), negativa
W = q * LT  # carga total (negativa)


def _beam_center_spring(ky: float | None, rigid_center: bool = False) -> Model:
    """Viga 0—1—2 articulada / móvil, q uniforme; resorte ky (o apoyo móvil) en el centro."""
    m = Model(
        nodes=[(0.0, 0.0), (A_SPAN, 0.0), (LT, 0.0)],
        members=[Member(0, 1, SEC, MAT), Member(1, 2, SEC, MAT)],
        supports=[Support(0, ST.PINNED), Support(2, ST.ROLLER)],
        distributed_loads=[DistributedLoad(0, q, q), DistributedLoad(1, q, q)],
    )
    if rigid_center:
        m.supports.append(Support(1, ST.ROLLER))
    if ky is not None:
        m.springs = [Spring(1, ky=ky)]
    return m


@pytest.mark.parametrize("ratio", [0.0, 0.1, 1.0, 10.0])  # ky / (48EI/L³)
def test_beam_central_spring_exact(ratio):
    """Compatibilidad en el centro: δ = δ₀ / (1 + ky·L³/48EI), δ₀ = 5qL⁴/384EI ; R_resorte = −ky·δ."""
    ky = ratio * K_BEAM
    r = solve(_beam_center_spring(ky))
    delta = DELTA0 / (1.0 + ratio)
    Rs = -ky * delta
    assert r.displacements[1, 1] == pytest.approx(delta, rel=REL)
    assert r.reactions[1][1] == pytest.approx(Rs, rel=REL, abs=1e-9 * abs(W))
    assert r.reactions[0][1] == pytest.approx(0.5 * (-W - Rs), rel=REL)
    assert r.reactions[2][1] == pytest.approx(0.5 * (-W - Rs), rel=REL)
    # Momento en el centro: viga simple (−q·L²/8 > 0) menos el efecto de la reacción Rs·L/4
    assert r.members[0].M[-1] == pytest.approx(-q * LT**2 / 8 - Rs * LT / 4, rel=REL)


def test_beam_central_spring_zero_equals_simple_beam():
    """ky = 0: viga simple exacta (δ = 5qL⁴/384EI, M = qL²/8, R = qL/2)."""
    r_spring, r_simple = solve(_beam_center_spring(0.0)), solve(_beam_center_spring(None))
    assert np.array_equal(r_spring.displacements, r_simple.displacements)
    assert r_spring.displacements[1, 1] == pytest.approx(DELTA0, rel=REL)


def test_beam_central_spring_convergence_limits():
    """ky → 0: viga simple ; ky → ∞: viga continua de 2 tramos (R_c = 5qL/8, R_ext = 3qL/16, M_c = qa²/8).

    Error relativo del límite ≈ 1/(1 + ky/K_viga): tolerancia 1e-5 con ky/K_viga = 1e6 (y 1e-6).
    Además la convergencia es monótona en toda la serie.
    """
    ratios = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    res = [solve(_beam_center_spring(rt * K_BEAM)) for rt in ratios]
    uy = [abs(r.displacements[1, 1]) for r in res]
    rc = [r.reactions[1][1] for r in res]
    assert all(a > b for a, b in zip(uy, uy[1:], strict=False))
    assert all(a < b for a, b in zip(rc, rc[1:], strict=False))

    soft, stiff = res[0], res[-1]
    assert soft.displacements[1, 1] == pytest.approx(DELTA0, rel=1e-5)
    assert soft.members[0].M[-1] == pytest.approx(-q * LT**2 / 8, rel=1e-5)
    assert soft.reactions[0][1] == pytest.approx(-W / 2, rel=1e-5)

    assert stiff.reactions[1][1] == pytest.approx(-5 * W / 8, rel=1e-5)
    assert stiff.reactions[0][1] == pytest.approx(-3 * W / 16, rel=1e-5)
    assert stiff.members[0].M[-1] == pytest.approx(q * A_SPAN**2 / 8, rel=1e-5)  # hogging (−)
    assert abs(stiff.displacements[1, 1]) < 1e-5 * abs(DELTA0)

    cont = solve(_beam_center_spring(None, rigid_center=True))  # viga continua del solver
    assert stiff.reactions[1][1] == pytest.approx(cont.reactions[1][1], rel=1e-5)
    assert stiff.members[0].M[-1] == pytest.approx(cont.members[0].M[-1], rel=1e-5)


# --------------------------------------------------------------------------- #
# Test 3: equilibrio estático con reacciones de resortes
# --------------------------------------------------------------------------- #

H, B = 4000.0, 6000.0
SPRINGS_L = Spring(0, kx=2.0e4, ky=5.0e5, krz=3.0e10)
SPRINGS_R = Spring(3, kx=8.0e3, ky=2.0e5, krz=1.0e10)


def _portal_on_springs() -> Model:
    """Pórtico sin vínculos rígidos: bases sobre resortes kx, ky, krz distintos (asimétrico)."""
    return Model(
        nodes=[(0.0, 0.0), (0.0, H), (B, H), (B, 0.0)],
        members=[Member(0, 1, COL, MAT), Member(1, 2, SEC, MAT), Member(2, 3, COL, MAT)],
        springs=[SPRINGS_L, SPRINGS_R],
        nodal_loads=[NodalLoad(1, Fx=5_000.0), NodalLoad(2, Fy=-P, Mz=2.0e6)],
        distributed_loads=[DistributedLoad(1, q, q)],
    )


def _beam_mixed() -> Model:
    m = _beam_center_spring(1.0 * K_BEAM)
    m.springs.append(Spring(0, kx=1e4, krz=1e11))  # resorte sobre nodo con apoyo articulado
    m.nodal_loads = [NodalLoad(1, Fy=-P), NodalLoad(2, Mz=-3.0e6)]
    return m


MODELS: dict[str, Callable[[], Model]] = {
    "portico_sobre_resortes": _portal_on_springs,
    "viga_apoyos_mixtos": _beam_mixed,
    "voladizo_krz": lambda: _cantilever(3, 2.0 * EI / LC),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_global_equilibrium_with_spring_reactions(name):
    """ΣFx = ΣFy = ΣMz = 0 con cargas + reacciones (rígidas y −k·u de resortes); residuo < 1e-12."""
    m = MODELS[name]()
    r = solve(m)
    assert set(r.reactions) == {s.node for s in m.supports} | {s.node for s in m.springs}
    rx, ry, rm = _equilibrium_residual(m, r)
    assert rx < RESIDUAL and ry < RESIDUAL and rm < RESIDUAL


def test_portal_spring_reactions_are_minus_k_times_u():
    """En nodos sólo con resortes: (Rx, Ry, Mz) = −(kx·ux, ky·uy, krz·rz), con signo opuesto a u."""
    m = _portal_on_springs()
    r = solve(m)
    for sp in m.springs:
        u = r.displacements[sp.node]
        expected = -np.asarray(sp.stiffness) * u
        got = np.asarray(r.reactions[sp.node])
        assert np.array_equal(got, expected)  # asignada exacta
        assert np.all(np.sign(got) == -np.sign(u))  # signo opuesto al desplazamiento / giro
    # ΣRx equilibra la única carga horizontal (5 kN en +X)
    assert sum(v[0] for v in r.reactions.values()) == pytest.approx(-5_000.0, rel=1e-12)


def test_springs_only_without_horizontal_stiffness_is_mechanism():
    """Pórtico sobre resortes verticales y rotacionales pero kx = 0 en ambas bases: mecanismo."""
    m = _portal_on_springs()
    m.springs = [Spring(0, ky=5e5, krz=3e10), Spring(3, ky=2e5, krz=1e10)]
    with pytest.raises(StructuralError):
        solve(m)


def test_rotational_spring_restores_stiffness_at_fully_hinged_node():
    """Nudo con todas las barras articuladas + krz: θ = M₀/krz, Mz_reacción = −M₀, barras sin momento.

    Sin el resorte el mismo caso lanza StructuralError (no hay rigidez a giro en el nudo).
    """
    L = 3000.0
    M0, krz = 5.0e6, 1.0e10
    m = Model(
        nodes=[(0.0, 0.0), (L, 0.0), (2 * L, 0.0)],
        members=[Member(0, 1, SEC, MAT, release_end=True), Member(1, 2, SEC, MAT, release_start=True)],
        supports=[Support(0, ST.FIXED), Support(2, ST.FIXED)],
        nodal_loads=[NodalLoad(1, Mz=M0)],
    )
    with pytest.raises(StructuralError):
        solve(m)
    m.springs = [Spring(1, krz=krz)]
    r = solve(m)
    assert r.displacements[1, 2] == pytest.approx(M0 / krz, rel=1e-12)
    assert r.reactions[1][2] == pytest.approx(-M0, rel=1e-12)
    assert max(float(np.max(np.abs(mr.M))) for mr in r.members) <= 1e-6 * M0
