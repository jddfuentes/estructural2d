"""Cargas térmicas (ΔT uniforme, deformación axial impuesta) y pórticos 2D de topología libre.

Formulación (ver `core/solver.py`): ε₀ = α·ΔT; barra con alargamiento impedido ⇒ N₀ = −E·A·α·ΔT;
N = EA·(u_j − u_i)/L − E·A·α·ΔT. Referencias: Gere & Weaver, "Analysis of Framed Structures", §4.10
(efectos de temperatura y deformaciones previas); Timoshenko & Gere, "Mechanics of Materials", §2.5.
Tolerancias rel = 1e-6 (AGENTS.md §7.1). Donde el valor exacto es 0, se usa una tolerancia absoluta
de redondeo relativa a la fuerza de fijación E·A·α·ΔT (1e-9·EAαΔT) o a la escala de longitudes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.builders import build_frame, build_portal_frame
from core.materials import MATERIALS, SECTIONS, Material, custom
from core.model import (
    COMB_1_2D_1_6L,
    COMB_1_4D,
    COMB_SERVICE_D_L,
    DistributedLoad,
    LoadCase,
    LoadDirection,
    Member,
    Model,
    NodalLoad,
    Spring,
    Support,
    ThermalLoad,
)
from core.model import SupportType as ST
from core.solver import Results, solve, solve_combinations, solve_load_cases
from core.verification import check_combinations, check_safety, member_stresses

E = 200_000.0  # [MPa]
ALPHA = 1.2e-5  # [1/°C]
MAT = Material("acero", E=E, Sy=250.0, Su=400.0)  # α por defecto = 1,2·10⁻⁵ 1/°C
BAR = custom("barra", A=2000.0, I=8.0e6, c=50.0)
COL = custom("col", A=5000.0, I=5.0e7, c=150.0)
BEAM = custom("viga", A=4000.0, I=8.0e7, c=200.0)
L = 6000.0  # [mm]


def _bar(n_segments: int, left: ST, right: ST, delta_t: float, angle_deg: float = 0.0) -> Model:
    """Barra recta de longitud L a `angle_deg`, partida en `n_segments` barras, con ΔT en todas."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    nodes = [(k * L / n_segments * c, k * L / n_segments * s) for k in range(n_segments + 1)]
    members = [Member(k, k + 1, BAR, MAT) for k in range(n_segments)]
    supports = [Support(0, left)] + ([Support(n_segments, right)] if right is not ST.FREE else [])
    thermal = [ThermalLoad(k, delta_t) for k in range(n_segments)]
    return build_frame(nodes, members, supports, thermal_loads=thermal)


def _sum_reactions(model: Model, res: Results) -> tuple[float, float, float]:
    """ΣRx, ΣRy y ΣM respecto del origen de las reacciones (sin cargas aplicadas deben ser 0)."""
    fx = fy = mz = 0.0
    for n, (rx, ry, m) in res.reactions.items():
        x, y = model.nodes[n]
        fx, fy, mz = fx + rx, fy + ry, mz + m + x * ry - y * rx
    return fx, fy, mz


# --------------------------------------------------------------------------- #
# Material y modelo
# --------------------------------------------------------------------------- #


def test_material_alpha_default_and_validation():
    """α por defecto 1,2·10⁻⁵ 1/°C (acero, EN 1993-1-1 §3.2.6); aluminio 23,6·10⁻⁶ 1/°C."""
    assert MAT.alpha == 1.2e-5
    assert MATERIALS["ASTM A36"].alpha == 1.2e-5
    assert MATERIALS["Al 6061-T6"].alpha == pytest.approx(2.36e-5)
    for bad in (-1e-6, math.nan, math.inf):
        with pytest.raises(ValueError, match="dilatación"):
            Material("x", E=E, Sy=250.0, Su=400.0, alpha=bad)


def test_thermal_load_validation_and_load_cases():
    """ThermalLoad: hipótesis D por defecto, acepta "L", ε₀ = α·ΔT + ε_imp; barra inexistente ⇒ ValueError."""
    t = ThermalLoad(0, 30.0)
    assert t.case is LoadCase.DEAD
    assert ThermalLoad(0, 30.0, "L").case is LoadCase.LIVE
    assert ThermalLoad(0, 30.0, strain=1e-4).imposed_strain(ALPHA) == pytest.approx(ALPHA * 30.0 + 1e-4)
    with pytest.raises(ValueError, match="finito"):
        ThermalLoad(0, math.nan)
    with pytest.raises(ValueError, match="Hipótesis"):
        ThermalLoad(0, 10.0, "W")
    m = _bar(1, ST.PINNED, ST.ROLLER, 0.0)
    m.thermal_loads = [ThermalLoad(0, 10.0, LoadCase.LIVE)]
    assert m.load_cases() == (LoadCase.LIVE,) and m.has_live_loads()
    m.thermal_loads = [ThermalLoad(3, 10.0)]
    with pytest.raises(ValueError, match="térmica en barra inexistente"):
        solve(m)


# --------------------------------------------------------------------------- #
# Test 1 — barra libre de dilatarse: ΔL = α·ΔT·L, N = 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_segments", [1, 3])
def test_free_bar_pinned_roller_elongates_without_stress(n_segments):
    """Articulado + móvil, ΔT = +50 °C: ΔL = α·ΔT·L = 3,6 mm, N = 0, M = 0, reacciones nulas."""
    dT = 50.0
    m = _bar(n_segments, ST.PINNED, ST.ROLLER, dT)
    r = solve(m)
    dL = ALPHA * dT * L
    assert dL == pytest.approx(3.6)
    assert r.displacements[-1, 0] == pytest.approx(dL, rel=1e-6)
    assert np.allclose(r.displacements[:, 1], 0.0, atol=1e-12)
    n0 = E * BAR.A * ALPHA * dT
    for mr in r.members:
        assert np.allclose(mr.N, 0.0, atol=1e-9 * n0)
        assert np.allclose(mr.M, 0.0, atol=1e-9 * n0 * L)
        # elástica axial u(x) = α·ΔT·(x0 + x): lineal y continua entre barras
        x0 = m.nodes[m.members[mr.member].i][0]
        assert mr.displacement[:, 0] == pytest.approx(ALPHA * dT * (x0 + mr.x), rel=1e-6, abs=1e-12)
    for rx, ry, mz in r.reactions.values():
        assert abs(rx) < 1e-9 * n0 and abs(ry) < 1e-9 * n0 and abs(mz) < 1e-9 * n0 * L
    assert check_safety(m, r).sigma_max == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("angle", [30.0, 135.0, -60.0])
def test_inclined_cantilever_free_thermal_strain(angle):
    """Ménsula inclinada (empotrada en i, libre en j), ΔT = −40 °C: δ_punta = α·ΔT·L·(cos θ, sin θ), N = 0."""
    dT = -40.0
    m = _bar(2, ST.FIXED, ST.FREE, dT, angle)
    r = solve(m)
    d = ALPHA * dT * L
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    assert r.displacements[-1, 0] == pytest.approx(d * c, rel=1e-6)
    assert r.displacements[-1, 1] == pytest.approx(d * s, rel=1e-6)
    assert r.displacements[-1, 2] == pytest.approx(0.0, abs=1e-15)
    n0 = E * BAR.A * ALPHA * abs(dT)
    assert all(np.allclose(mr.N, 0.0, atol=1e-9 * n0) for mr in r.members)


# --------------------------------------------------------------------------- #
# Test 2 — barra biempotrada: u = 0, N = −E·A·α·ΔT, R = E·A·α·ΔT
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n_segments", "angle"), [(1, 0.0), (4, 0.0), (2, 45.0), (1, 90.0)])
def test_fixed_fixed_bar_restrained_expansion(n_segments, angle):
    """Biempotrada, ΔT = +30 °C: desplazamientos nulos, N = −EAαΔT (compresión), σ = −E·α·ΔT = −72 MPa,
    reacción axial R = EAαΔT empujando hacia la barra en ambos extremos; ΣR = 0."""
    dT = 30.0
    m = _bar(n_segments, ST.FIXED, ST.FIXED, dT, angle)
    r = solve(m)
    R = E * BAR.A * ALPHA * dT
    assert R == pytest.approx(144_000.0)
    assert np.allclose(r.displacements, 0.0, atol=1e-12)
    for mr in r.members:
        assert mr.N == pytest.approx(np.full_like(mr.N, -R), rel=1e-6)
        assert np.allclose(mr.M, 0.0, atol=1e-9 * R * L)
        assert np.allclose(mr.displacement, 0.0, atol=1e-12)
        assert mr.sigma_top == pytest.approx(np.full_like(mr.x, -E * ALPHA * dT), rel=1e-6)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    r0, r1 = r.reactions[0], r.reactions[n_segments]
    # el vínculo en i empuja hacia +x local, el de j hacia −x local (barra comprimida)
    assert (r0[0], r0[1]) == pytest.approx((R * c, R * s), rel=1e-6, abs=1e-9 * R)
    assert (r1[0], r1[1]) == pytest.approx((-R * c, -R * s), rel=1e-6, abs=1e-9 * R)
    assert _sum_reactions(m, r) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9 * R * L)
    assert r.extreme("N").value == pytest.approx(-R, rel=1e-6)


def test_bar_pinned_both_ends_and_strut_transmit_thermal_force():
    """Biela (rótulas en ambos extremos) entre dos articulados: la dilatación impedida da el mismo
    N = −EAαΔT; con rótulas y ambos extremos fijos el axil se transmite íntegro."""
    dT = 30.0
    R = E * BAR.A * ALPHA * dT
    for supports, member in (
        ((ST.PINNED, ST.PINNED), Member(0, 1, BAR, MAT)),
        ((ST.FIXED, ST.FIXED), Member(0, 1, BAR, MAT, release_start=True, release_end=True)),
    ):
        m = build_frame([(0.0, 0.0), (L, 0.0)], [member], [Support(0, supports[0]), Support(1, supports[1])],
                        thermal_loads=[ThermalLoad(0, dT)])
        r = solve(m)
        assert r.members[0].N == pytest.approx(np.full_like(r.members[0].x, -R), rel=1e-6)
        assert r.reactions[0][0] == pytest.approx(R, rel=1e-6)
        assert np.allclose(r.members[0].M, 0.0, atol=1e-9 * R * L)


def test_imposed_strain_equivalent_to_temperature():
    """Deformación axial impuesta ε_imp ≡ α·ΔT: ThermalLoad(m, 0, strain=α·ΔT) da la misma respuesta."""
    dT = 25.0
    m1 = _bar(2, ST.FIXED, ST.ROLLER, dT, 20.0)
    m2 = _bar(2, ST.FIXED, ST.ROLLER, 0.0, 20.0)
    m2.thermal_loads = [ThermalLoad(k, 0.0, strain=ALPHA * dT) for k in range(2)]
    r1, r2 = solve(m1), solve(m2)
    assert r2.displacements == pytest.approx(r1.displacements, rel=1e-9, abs=1e-12)
    for a, b in zip(r1.members, r2.members, strict=True):
        assert b.N == pytest.approx(a.N, rel=1e-9, abs=1e-6)
        assert b.M == pytest.approx(a.M, rel=1e-9, abs=1e-3)


def test_thermal_stresses_in_safety_check():
    """check_safety: biempotrada ΔT = +30 °C ⇒ σ_max = σ_vM = E·α·ΔT = 72 MPa, FS = Sy/72."""
    m = _bar(1, ST.FIXED, ST.FIXED, 30.0)
    r = solve(m)
    sc = check_safety(m, r)
    assert sc.sigma_max == pytest.approx(72.0, rel=1e-6)
    assert sc.sigma_vm_max == pytest.approx(72.0, rel=1e-6)
    assert sc.fs == pytest.approx(250.0 / 72.0, rel=1e-6)
    st = member_stresses(BAR, r.members[0])
    assert st.sigma_axial == pytest.approx(np.full_like(st.x, -72.0), rel=1e-6)


def test_thermal_compression_enters_buckling_interaction():
    """stability=True: la compresión térmica de una columna biempotrada entra en H1 con N_comp = EAαΔT."""
    ipe = SECTIONS["IPE"]["IPE 200"]
    H, dT = 3000.0, 40.0
    m = build_frame([(0.0, 0.0), (0.0, H)], [Member(0, 1, ipe, MAT)],
                    [Support(0, ST.FIXED), Support(1, ST.FIXED)], thermal_loads=[ThermalLoad(0, dT)])
    r = solve(m)
    P = E * ipe.A * ALPHA * dT
    sc = check_safety(m, r, stability=True, effective_length_factor_k=0.5)
    assert sc.stability[0].N_comp == pytest.approx(P, rel=1e-6)
    assert sc.stability_utilization > 0.0
    cc = check_combinations(m, stability=True, effective_length_factor_k=0.5)
    comb, sc_u = cc.stability_governing
    assert comb == COMB_1_4D
    assert sc_u.stability[0].N_comp == pytest.approx(1.4 * P, rel=1e-6)


# --------------------------------------------------------------------------- #
# Test 3 — pórtico portal con ΔT en el dintel
# --------------------------------------------------------------------------- #

S, H = 8000.0, 4000.0


def _portal_thermal(dT: float, base: ST = ST.FIXED) -> Model:
    m = build_portal_frame(S, H, COL, BEAM, MAT, base, base)
    m.thermal_loads = [ThermalLoad(1, dT)]  # barra 1 = dintel (nodos 1 → 2)
    return m


def test_portal_beam_thermal_pushes_columns_outward():
    """ΔT = +40 °C en el dintel de un pórtico biempotrado: los nudos se abren simétricamente
    (0 < Δ < α·ΔT·S), las columnas flexan con momentos de base opuestos, el dintel queda comprimido y
    ΣFx = ΣFy = ΣM = 0 (cargas autoequilibradas)."""
    dT = 40.0
    m = _portal_thermal(dT)
    r = solve(m)
    ux_l, ux_r = r.displacements[1, 0], r.displacements[2, 0]
    assert ux_l < 0.0 < ux_r
    assert ux_l == pytest.approx(-ux_r, rel=1e-6)
    spread = ux_r - ux_l
    assert 0.0 < spread < ALPHA * dT * S
    # columnas flexadas: momentos en las bases no nulos, iguales y opuestos (simetría)
    m_base_l, m_base_r = r.reactions[0][2], r.reactions[3][2]
    assert abs(m_base_l) > 1e3
    assert m_base_l == pytest.approx(-m_base_r, rel=1e-6)
    # empuje hacia afuera: las bases reaccionan hacia adentro
    assert r.reactions[0][0] > 0.0 > r.reactions[3][0]
    # dintel comprimido; el empuje en las bases equilibra su compresión
    n_beam = r.members[1].N
    assert np.all(n_beam < 0.0)
    assert -n_beam[0] == pytest.approx(r.reactions[0][0], rel=1e-6)
    P0 = E * BEAM.A * ALPHA * dT
    assert _sum_reactions(m, r) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9 * P0 * S)


def test_portal_thermal_equals_equivalent_nodal_loads():
    """Principio de equivalencia (Gere & Weaver §4.10): la respuesta a ΔT es la de las cargas nodales
    equivalentes (−EAαΔT en i, +EAαΔT en j, ejes de la barra) más el estado de empotramiento perfecto
    del dintel (N₀ = −EAαΔT): mismos desplazamientos y momentos; N_dintel = N_nodal − EAαΔT."""
    dT = 40.0
    P0 = E * BEAM.A * ALPHA * dT
    m_t = _portal_thermal(dT)
    m_f = build_portal_frame(S, H, COL, BEAM, MAT, ST.FIXED, ST.FIXED)
    m_f.nodal_loads = [NodalLoad(1, Fx=-P0), NodalLoad(2, Fx=P0)]
    r_t, r_f = solve(m_t), solve(m_f)
    assert r_t.displacements == pytest.approx(r_f.displacements, rel=1e-6, abs=1e-12)
    for k in range(3):
        assert r_t.members[k].M == pytest.approx(r_f.members[k].M, rel=1e-6, abs=1e-3)
    assert r_t.members[1].N == pytest.approx(r_f.members[1].N - P0, rel=1e-6)
    assert r_t.members[0].N == pytest.approx(r_f.members[0].N, rel=1e-6, abs=1e-6)


def test_cantilever_columns_with_heated_strut_closed_form():
    """Dos columnas en ménsula (k_c = 3EI/H³) unidas por una biela calentada: compatibilidad
    2P/k_c = α·ΔT·S − P·S/(EA) ⇒ P = α·ΔT·S / (2/k_c + S/(EA)); M_base = P·H; N_biela = −P."""
    dT = 60.0
    nodes = [(0.0, 0.0), (0.0, H), (S, H), (S, 0.0)]
    members = [
        Member(0, 1, COL, MAT),
        Member(1, 2, BEAM, MAT, release_start=True, release_end=True),
        Member(3, 2, COL, MAT),
    ]
    m = build_frame(nodes, members, [Support(0, ST.FIXED), Support(3, ST.FIXED)],
                    thermal_loads=[ThermalLoad(1, dT)])
    r = solve(m)
    k_c = 3 * E * COL.I / H**3
    P = ALPHA * dT * S / (2 / k_c + S / (E * BEAM.A))
    assert r.members[1].N == pytest.approx(np.full_like(r.members[1].x, -P), rel=1e-6)
    assert r.displacements[1, 0] == pytest.approx(-P / k_c, rel=1e-6)
    assert abs(r.reactions[0][2]) == pytest.approx(P * H, rel=1e-6)
    assert r.members[1].M == pytest.approx(np.zeros_like(r.members[1].x), abs=1e-9 * P * H)
    assert _sum_reactions(m, r) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9 * P * S)


# --------------------------------------------------------------------------- #
# Pórticos genéricos (build_frame)
# --------------------------------------------------------------------------- #


def test_build_frame_reproduces_portal_builder():
    """build_frame con la misma topología que build_portal_frame da resultados idénticos."""
    ref = build_portal_frame(S, H, COL, BEAM, MAT, ST.FIXED, ST.PINNED, beam_q=-20.0, lateral_load=15_000.0)
    m = build_frame(ref.nodes, ref.members, ref.supports, ref.nodal_loads, ref.distributed_loads)
    r0, r1 = solve(ref), solve(m)
    assert r1.displacements == pytest.approx(r0.displacements, rel=1e-12, abs=1e-15)
    for a, b in zip(r0.members, r1.members, strict=True):
        assert b.M == pytest.approx(a.M, rel=1e-12, abs=1e-6)


def test_truss_triangle_isostatic_thermal_and_gravity():
    """Cercha triangular isostática de bielas (articulado + móvil), luz S, flecha h, carga P en cumbrera:
    pares N = −P/(2·sin θ), tirante N = P/(2·tan θ). ΔT en el tirante (isostática) no altera N y mueve la
    cumbrera ux = δ/2, uy = −(S/2)·δ/(2h) con δ = α·ΔT·S (compatibilidad lineal, pares sin alargamiento)."""
    h, Pv, dT = 2000.0, -50_000.0, 35.0
    nodes = [(0.0, 0.0), (S / 2, h), (S, 0.0)]
    members = [Member(a, b, BAR, MAT, True, True) for a, b in ((0, 1), (1, 2), (0, 2))]
    supports = [Support(0, ST.PINNED), Support(2, ST.ROLLER)]
    th = math.atan2(h, S / 2)

    m_p = build_frame(nodes, members, supports, nodal_loads=[NodalLoad(1, Fy=Pv)])
    r_p = solve(m_p)
    P = -Pv
    for k in (0, 1):
        assert r_p.members[k].N[0] == pytest.approx(-P / (2 * math.sin(th)), rel=1e-6)
    assert r_p.members[2].N[0] == pytest.approx(P / (2 * math.tan(th)), rel=1e-6)

    m_t = build_frame(nodes, members, supports, thermal_loads=[ThermalLoad(2, dT)])
    r_t = solve(m_t)
    n0 = E * BAR.A * ALPHA * dT
    for mr in r_t.members:
        assert np.allclose(mr.N, 0.0, atol=1e-9 * n0)
    delta = ALPHA * dT * S
    assert r_t.displacements[2, 0] == pytest.approx(delta, rel=1e-6)
    assert r_t.displacements[1, 0] == pytest.approx(delta / 2, rel=1e-6)
    assert r_t.displacements[1, 1] == pytest.approx(-(S / 2) * delta / (2 * h), rel=1e-6)


def test_gable_frame_inclined_rafters_equilibrium():
    """Pórtico a dos aguas (pares inclinados, carga local-y, carga global-y, ΔT en un par, resorte):
    equilibrio global ΣF = 0 y ΣM = 0 con las cargas aplicadas."""
    hc, hr = 4000.0, 1500.0
    nodes = [(0.0, 0.0), (0.0, hc), (S / 2, hc + hr), (S, hc), (S, 0.0)]
    members = [
        Member(0, 1, COL, MAT), Member(1, 2, BEAM, MAT), Member(2, 3, BEAM, MAT), Member(4, 3, COL, MAT),
    ]
    q, w = -10.0, 3.0  # [N/mm]
    m = build_frame(
        nodes, members, [Support(0, ST.FIXED), Support(4, ST.PINNED)],
        nodal_loads=[NodalLoad(2, Fx=8000.0, Fy=-20_000.0, Mz=5e6)],
        distributed_loads=[DistributedLoad(1, q, q), DistributedLoad(2, w, w, LoadDirection.LOCAL_Y)],
        thermal_loads=[ThermalLoad(1, 50.0)],
        springs=[Spring(4, krz=1e9)],
    )
    r = solve(m)
    # resultante de cargas aplicadas y su momento respecto del origen
    fx, fy, mz = 8000.0, -20_000.0, 5e6 + (S / 2) * -20_000.0 - (hc + hr) * 8000.0
    L1 = m.member_length(1)
    fy += q * L1
    mz += q * L1 * (S / 4)  # resultante en el punto medio del par izquierdo
    c, s = m.member_cos_sin(2)
    L2 = m.member_length(2)
    gx, gy = -s * w * L2, c * w * L2  # y local = x rotado +90°
    xm, ym = 3 * S / 4, hc + hr / 2
    fx, fy, mz = fx + gx, fy + gy, mz + xm * gy - ym * gx
    rx, ry, rm = _sum_reactions(m, r)
    scale = abs(fy) * S
    assert rx + fx == pytest.approx(0.0, abs=1e-8 * abs(fy))
    assert ry + fy == pytest.approx(0.0, abs=1e-8 * abs(fy))
    assert rm + mz == pytest.approx(0.0, abs=1e-8 * scale)


def test_build_frame_validation_errors():
    """build_frame rechaza nodos coincidentes, nodos sueltos, barras duplicadas y nodos inválidos."""
    sup = [Support(0, ST.FIXED)]
    with pytest.raises(ValueError, match="coincidentes"):
        build_frame([(0.0, 0.0), (0.0, 0.0), (1000.0, 0.0)], [Member(0, 2, BAR, MAT)], sup)
    with pytest.raises(ValueError, match="sin barras"):
        build_frame([(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)], [Member(0, 1, BAR, MAT)], sup)
    with pytest.raises(ValueError, match="mismos nodos"):
        build_frame([(0.0, 0.0), (1000.0, 0.0)], [Member(0, 1, BAR, MAT), Member(1, 0, BAR, MAT)], sup)
    with pytest.raises(ValueError, match="nodos inválidos"):
        build_frame([(0.0, 0.0), (1000.0, 0.0)], [Member(0, 1, BAR, MAT), Member(1, 5, BAR, MAT)], sup)
    with pytest.raises(ValueError, match="no finitas"):
        build_frame([(0.0, 0.0), (math.nan, 0.0)], [Member(0, 1, BAR, MAT)], sup)


def test_solve_does_not_mutate_thermal_model():
    """solve() es pura también con cargas térmicas (AGENTS.md §6)."""
    m = _portal_thermal(30.0)
    before = repr(m)
    solve(m)
    solve_load_cases(m)
    assert repr(m) == before


# --------------------------------------------------------------------------- #
# Test 4 — combinaciones de carga con temperatura
# --------------------------------------------------------------------------- #


def _portal_loads(gD: float, gL: float, split: bool) -> Model:
    """Pórtico con q_D, ΔT_D (dintel) y q_L. split=False: cargas ya mayoradas, todas en D."""
    m = build_portal_frame(S, H, COL, BEAM, MAT, ST.FIXED, ST.PINNED)
    qD, qL, dT = -15.0, -10.0, 35.0
    if split:
        m.distributed_loads = [DistributedLoad(1, qD, qD), DistributedLoad(1, qL, qL, case=LoadCase.LIVE)]
        m.thermal_loads = [ThermalLoad(1, dT)]
    else:
        q = gD * qD + gL * qL
        m.distributed_loads = [DistributedLoad(1, q, q)]
        m.thermal_loads = [ThermalLoad(1, gD * dT)]
    return m


def _assert_same(a: Results, b: Results) -> None:
    assert a.displacements == pytest.approx(b.displacements, rel=1e-6, abs=1e-9)
    for n in b.reactions:
        assert a.reactions[n] == pytest.approx(b.reactions[n], rel=1e-6, abs=1e-3)
    for ma, mb in zip(a.members, b.members, strict=True):
        assert ma.N == pytest.approx(mb.N, rel=1e-6, abs=1e-3)
        assert ma.M == pytest.approx(mb.M, rel=1e-6, abs=1.0)


@pytest.mark.parametrize("comb", [COMB_1_4D, COMB_1_2D_1_6L, COMB_SERVICE_D_L], ids=lambda c: c.name)
def test_combinations_with_thermal_dead_load(comb):
    """γ_D·(q_D + ΔT) + γ_L·q_L por superposición ≡ resolver con q y ΔT ya mayorados
    (1.4D, 1.2D + 1.6L y 1.0D + 1.0L)."""
    split = _portal_loads(1.0, 1.0, split=True)
    (res,) = solve_combinations(split, [comb])
    _assert_same(res, solve(_portal_loads(comb.gamma_D, comb.gamma_L, split=False)))


def test_thermal_as_live_load_and_uls_check():
    """ΔT asignada a L: 1.4D no la incluye; en 1.2D + 1.6L gobierna σ_u = 1.6·E·α·ΔT (biempotrada)."""
    m = _bar(1, ST.FIXED, ST.FIXED, 0.0)
    m.thermal_loads = [ThermalLoad(0, 30.0, LoadCase.LIVE)]
    cases = solve_load_cases(m)
    assert set(cases) == {LoadCase.DEAD, LoadCase.LIVE}
    assert np.allclose(cases[LoadCase.DEAD].members[0].N, 0.0)
    cc = check_combinations(m)
    sigma = E * ALPHA * 30.0
    assert cc.governing == COMB_1_2D_1_6L
    assert cc.sigma_vm_max == pytest.approx(1.6 * sigma, rel=1e-6)
    assert cc.utilization == pytest.approx(1.6 * sigma / (0.9 * 250.0), rel=1e-6)
    by_name = {c.name: sc for c, sc in cc.strength_checks}
    assert by_name["1.4D"].sigma_max == pytest.approx(0.0, abs=1e-9)


def test_thermal_dead_load_uls_check():
    """ΔT en D (biempotrada): gobierna 1.4D con σ_u = 1.4·E·α·ΔT = 100,8 MPa ≤ φ·Sy = 225 MPa."""
    m = _bar(1, ST.FIXED, ST.FIXED, 30.0)
    cc = check_combinations(m)
    assert cc.governing == COMB_1_4D
    assert cc.sigma_vm_max == pytest.approx(100.8, rel=1e-6)
    assert cc.ok
