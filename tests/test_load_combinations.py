"""Hipótesis de carga D / L y combinaciones CIRSOC 301-2005 §A.4.2 / AISC 360-16 §B2 (LRFD).

Unidades: mm, N, MPa. Viga simplemente apoyada de luz L (Gere, "Mecánica de materiales", tablas
de vigas):
- carga uniforme q:          M_máx = qL²/8,           δ_máx = 5qL⁴/(384EI)
- puntual P en el centro:    M_máx = PL/4,             δ_máx = PL³/(48EI)
Superposición lineal (elástico lineal): R(γ_D·D + γ_L·L) = γ_D·R(D) + γ_L·R(L).
"""

from __future__ import annotations

import numpy as np
import pytest

import core.solver as solver_mod
from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam
from core.materials import Material, custom
from core.model import (
    COMB_1_2D_1_6L,
    COMB_1_4D,
    COMB_SERVICE_D_L,
    DistributedLoad,
    LoadCase,
    LoadCombination,
    NodalLoad,
)
from core.model import SupportType as ST
from core.solver import Results, combine_results, solve, solve_combinations, solve_load_cases
from core.verification import PHI_DEFAULT, Status, check_combinations

REL = 1e-6  # AGENTS.md §7.1
RTOL = 1e-9  # superposición: misma aritmética salvo redondeo

E = 200_000.0  # MPa
I = 2.0e7  # mm⁴  # noqa: E741
A = 3000.0  # mm²
C = 100.0  # mm (sección simétrica)
L = 6000.0  # mm
SY = 250.0  # MPa

MAT = Material("test", E=E, Sy=SY, Su=400.0)
SEC = custom("test", A=A, I=I, c=C)
SIMPLE = [BeamSupport(0.0, ST.PINNED), BeamSupport(L, ST.ROLLER)]


def beam(q_D: float = 0.0, P_L: float = 0.0, x_P: float = L / 2, q_L: float = 0.0):
    """Viga simple: q_D uniforme (D), P_L puntual en x_P (L) y q_L uniforme (L)."""
    dist = [BeamDistLoad(0.0, L, q_D, q_D)] if q_D else []
    if q_L:
        dist.append(BeamDistLoad(0.0, L, q_L, q_L, case=LoadCase.LIVE))
    points = [BeamPointLoad(x_P, Fy=P_L, case=LoadCase.LIVE)] if P_L else []
    return build_beam(L, SIMPLE, SEC, MAT, points or None, dist or None)


def assert_same_results(a: Results, b: Results) -> None:
    """Igualdad punto por punto de dos `Results` (mismas estaciones y mismos valores)."""
    scale_u = max(np.abs(b.displacements).max(), 1e-300)
    np.testing.assert_allclose(a.displacements, b.displacements, rtol=RTOL, atol=RTOL * scale_u)
    assert a.reactions.keys() == b.reactions.keys()
    for n in b.reactions:
        np.testing.assert_allclose(a.reactions[n], b.reactions[n], rtol=RTOL, atol=1e-6)
    assert len(a.members) == len(b.members)
    for ma, mb in zip(a.members, b.members, strict=True):
        assert ma.x.shape == mb.x.shape, "las estaciones de muestreo deben coincidir"
        np.testing.assert_allclose(ma.x, mb.x, rtol=0.0, atol=RTOL * L)
        for attr in ("N", "V", "M", "u_local", "v_local", "displacement", "sigma_top", "sigma_bot",
                     "end_forces"):
            va, vb = getattr(ma, attr), getattr(mb, attr)
            np.testing.assert_allclose(va, vb, rtol=RTOL, atol=RTOL * max(np.abs(vb).max(), 1e-9),
                                       err_msg=f"barra {mb.member}, {attr}")


# --------------------------------------------------------------------------- #
# Modelo de datos y retrocompatibilidad
# --------------------------------------------------------------------------- #


def test_load_case_default_dead_and_string_coercion():
    """Sin `case` toda carga es D (retrocompatible); "L" se normaliza a LoadCase.LIVE."""
    assert NodalLoad(0, Fy=-1.0).case is LoadCase.DEAD
    assert DistributedLoad(0, -1.0, -1.0).case is LoadCase.DEAD
    assert NodalLoad(0, Fy=-1.0, case="L").case is LoadCase.LIVE  # type: ignore[arg-type]
    assert BeamPointLoad(0.0).case is LoadCase.DEAD and BeamDistLoad(0, 1, 0, 0).case is LoadCase.DEAD
    with pytest.raises(ValueError, match="Hipótesis de carga desconocida"):
        DistributedLoad(0, -1.0, -1.0, case="W")  # type: ignore[arg-type]


def test_load_combination_validation_and_standard_values():
    """Factores CIRSOC 301 (A.4-1), (A.4-2) y servicio D + L."""
    assert (COMB_1_4D.gamma_D, COMB_1_4D.gamma_L, COMB_1_4D.limit_state) == (1.4, 0.0, "ELU")
    assert (COMB_1_2D_1_6L.gamma_D, COMB_1_2D_1_6L.gamma_L) == (1.2, 1.6)
    assert (COMB_SERVICE_D_L.gamma_D, COMB_SERVICE_D_L.gamma_L, COMB_SERVICE_D_L.limit_state) == (
        1.0, 1.0, "ELS")
    with pytest.raises(ValueError, match="estado límite"):
        LoadCombination("x", 1.0, 1.0, "ULS")
    with pytest.raises(ValueError, match="γ_L"):
        LoadCombination("x", 1.0, -1.6)


def test_without_live_loads_behaves_like_solve():
    """Sin cargas L: sólo hipótesis D, idéntica a `solve`; 1.0D devuelve el mismo objeto."""
    model = beam(q_D=-10.0)
    assert not model.has_live_loads()
    cases = solve_load_cases(model)
    assert list(cases) == [LoadCase.DEAD]
    assert_same_results(cases[LoadCase.DEAD], solve(model))
    one = LoadCombination("1.0D", 1.0, 0.0, "ELS")
    r_1d, r_14d = solve_combinations(model, [one, COMB_1_4D])
    assert_same_results(r_1d, solve(model))
    assert r_14d.extreme("M").value == pytest.approx(1.4 * 10.0 * L**2 / 8, rel=REL)  # 1.4·qL²/8


def test_stiffness_assembled_once(monkeypatch):
    """K se ensambla una única vez para D, L y todas las combinaciones."""
    calls = []
    original = solver_mod._assemble

    def counting(model):
        calls.append(1)
        return original(model)

    monkeypatch.setattr(solver_mod, "_assemble", counting)
    solve_combinations(beam(q_D=-10.0, P_L=-20_000.0), [COMB_1_4D, COMB_1_2D_1_6L, COMB_SERVICE_D_L])
    assert len(calls) == 1


def test_solve_does_not_mutate_model_with_cases():
    model = beam(q_D=-10.0, P_L=-20_000.0)
    before = (list(model.nodal_loads), list(model.distributed_loads))
    solve_combinations(model, [COMB_1_2D_1_6L])
    assert (model.nodal_loads, model.distributed_loads) == before


# --------------------------------------------------------------------------- #
# Test 1: superposición exacta
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("x_P", [L / 2, 0.37 * L])
def test_superposition_matches_single_factored_run(x_P):
    """1.2D + 1.6L por superposición ≡ una corrida con 1.2·q_D y 1.6·P_L, punto por punto."""
    q_D, P_L = -12.0, -35_000.0
    model = beam(q_D=q_D, P_L=P_L, x_P=x_P)
    cases = solve_load_cases(model)
    assert set(cases) == {LoadCase.DEAD, LoadCase.LIVE}

    combined = combine_results(cases[LoadCase.DEAD], cases[LoadCase.LIVE], 1.2, 1.6)
    scaled = solve(build_beam(L, SIMPLE, SEC, MAT, [BeamPointLoad(x_P, Fy=1.6 * P_L)],
                              [BeamDistLoad(0.0, L, 1.2 * q_D, 1.2 * q_D)]))
    assert_same_results(combined, scaled)
    assert_same_results(solve_combinations(model, [COMB_1_2D_1_6L])[0], scaled)
    for qty in ("M", "V", "sigma", "deflection"):
        assert combined.extreme(qty).value == pytest.approx(scaled.extreme(qty).value, rel=RTOL)

    # γ = 1 en ambas hipótesis ≡ `solve` con todas las cargas (comportamiento previo)
    assert_same_results(combine_results(cases[LoadCase.DEAD], cases[LoadCase.LIVE], 1.0, 1.0),
                        solve(model))


def test_superposition_analytic_midspan():
    """Centro de luz: M_u = 1.2·qL²/8 + 1.6·PL/4 ; equilibrio ΣRy = −(1.2·qL + 1.6·P)."""
    q_D, P_L = -12.0, -35_000.0
    res = solve_combinations(beam(q_D=q_D, P_L=P_L), [COMB_1_2D_1_6L])[0]
    M_u = -(1.2 * q_D * L**2 / 8 + 1.6 * P_L * L / 4)  # > 0 (sagging)
    assert res.extreme("M").value == pytest.approx(M_u, rel=REL)
    assert sum(r[1] for r in res.reactions.values()) == pytest.approx(-(1.2 * q_D * L + 1.6 * P_L),
                                                                      rel=REL)


def test_combination_with_hinges_keeps_zero_moment():
    """Rótula interna: M = 0 exacto en el extremo liberado también en la combinación."""
    from core.model import Member, Model, Support

    n = [(0.0, 0.0), (3000.0, 0.0), (5000.0, 0.0), (8000.0, 0.0)]
    members = [Member(0, 1, SEC, MAT), Member(1, 2, SEC, MAT, release_end=True), Member(2, 3, SEC, MAT)]
    model = Model(nodes=n, members=members,
                  supports=[Support(0, ST.FIXED), Support(1, ST.ROLLER), Support(3, ST.PINNED)],
                  distributed_loads=[DistributedLoad(k, -8.0, -8.0) for k in range(3)],
                  nodal_loads=[NodalLoad(2, Fy=-10_000.0, case=LoadCase.LIVE)])
    res = solve_combinations(model, [COMB_1_2D_1_6L])[0]
    assert res.members[1].M[-1] == 0.0 and res.members[1].end_forces[5] == 0.0


def test_combine_requires_same_model():
    a = solve(beam(q_D=-10.0))
    b = solve(build_beam(L, SIMPLE + [BeamSupport(L / 2, ST.ROLLER)], SEC, MAT,
                         dist_loads=[BeamDistLoad(0.0, L, -10.0, -10.0)]))
    with pytest.raises(ValueError, match="modelos distintos"):
        combine_results(a, b, 1.2, 1.6)


# --------------------------------------------------------------------------- #
# Test 2: combinación gobernante
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("q_D", "q_L", "governing", "gammas"),
    [
        # L/D = 0.05 < 0.125 ⇒ 1.4·D > 1.2·D + 1.6·L
        (-20.0, -1.0, "1.4D", (1.4, 0.0)),
        # L/D = 2 ⇒ 1.2·D + 1.6·L = 22 > 1.4·D = 7 (N/mm)
        (-5.0, -10.0, "1.2D + 1.6L", (1.2, 1.6)),
    ],
)
def test_governing_uls_combination(q_D, q_L, governing, gammas):
    """σ_u = M_u·c/I con M_u = (γ_D·q_D + γ_L·q_L)·L²/8; gobierna la de mayor σ_u.

    Criterio 1.4D ↔ 1.2D + 1.6L: 1.4·D ≥ 1.2·D + 1.6·L  ⇔  L ≤ D/8.
    """
    chk = check_combinations(beam(q_D=q_D, q_L=q_L))
    assert chk.governing.name == governing
    g_D, g_L = gammas
    sigma_u = abs(g_D * q_D + g_L * q_L) * L**2 / 8 * C / I
    assert chk.strength.sigma_max == pytest.approx(sigma_u, rel=REL)
    assert chk.utilization == pytest.approx(sigma_u / (PHI_DEFAULT * SY), rel=REL)
    # la otra combinación da una tensión menor (la gobernante es el máximo de la envolvente)
    others = [c.sigma_max for comb, c in chk.strength_checks if comb.name != governing]
    assert others and max(others) < chk.strength.sigma_max
    assert chk.governing_results.extreme("sigma").value == pytest.approx(sigma_u, rel=REL)


def test_uls_status_and_issue_text():
    """σ_u > φ·Fy ⇒ ALERTA con la combinación nombrada; σ_u > Fy ⇒ FLUENCIA."""
    # σ_u(1.2D+1.6L) = (1.2·5 + 1.6·q_L)·L²/8·c/I; q_L tal que σ_u ≈ 0.95·Fy (> φ·Fy = 0.9·Fy)
    q_L = -((0.95 * SY * I / C * 8 / L**2) - 1.2 * 5.0) / 1.6
    chk = check_combinations(beam(q_D=-5.0, q_L=q_L), deflection_limit_ratio=0.0)
    assert chk.status is Status.ALERT and not chk.strength_ok
    assert chk.issues and chk.issues[0].startswith("ELU 1.2D + 1.6L")
    chk_fail = check_combinations(beam(q_D=-5.0, q_L=2 * q_L), deflection_limit_ratio=0.0)
    assert chk_fail.status is Status.FAIL


def test_check_combinations_rejects_bad_inputs():
    model = beam(q_D=-10.0)
    with pytest.raises(ValueError, match="φ"):
        check_combinations(model, phi=0.0)
    with pytest.raises(ValueError, match="no es de resistencia"):
        check_combinations(model, uls_combinations=[COMB_SERVICE_D_L])
    with pytest.raises(ValueError, match="no es de servicio"):
        check_combinations(model, service_combination=COMB_1_4D)


# --------------------------------------------------------------------------- #
# Test 3: flecha de servicio sin mayorar
# --------------------------------------------------------------------------- #


def test_service_deflection_is_unfactored():
    """δ_ELS = 5·q_D·L⁴/(384EI) + P_L·L³/(48EI) con γ_D = γ_L = 1 (sin mayorar)."""
    q_D, P_L = -12.0, -35_000.0
    chk = check_combinations(beam(q_D=q_D, P_L=P_L))
    delta = 5 * abs(q_D) * L**4 / (384 * E * I) + abs(P_L) * L**3 / (48 * E * I)
    assert chk.service_combination.gamma_D == 1.0 and chk.service_combination.gamma_L == 1.0
    assert chk.delta_max == pytest.approx(delta, rel=REL)
    assert chk.service_results.extreme("uy").value == pytest.approx(-delta, rel=REL)
    # no coincide con la flecha de las cargas mayoradas
    delta_u = 1.2 * 5 * abs(q_D) * L**4 / (384 * E * I) + 1.6 * abs(P_L) * L**3 / (48 * E * I)
    assert chk.governing_results.extreme("deflection").value == pytest.approx(delta_u, rel=REL)
    assert chk.delta_max < delta_u
    # control L/N con la flecha de servicio
    assert chk.service_check.deflection_ratio == pytest.approx(L / delta, rel=REL)
    assert chk.deflection_ok is (L / delta >= 300.0)


def test_service_deflection_limit_triggers_alert():
    """Flecha de servicio > L/N ⇒ ALERTA con mensaje ELS aunque la resistencia verifique."""
    q_D, P_L = -2.0, -6_000.0
    delta = 5 * abs(q_D) * L**4 / (384 * E * I) + abs(P_L) * L**3 / (48 * E * I)  # ≈ 3.04 mm
    N = 0.5 * L / delta  # δ_adm = L/N = 2·δ ⇒ verifica
    assert check_combinations(beam(q_D=q_D, P_L=P_L), deflection_limit_ratio=N).ok
    chk = check_combinations(beam(q_D=q_D, P_L=P_L), deflection_limit_ratio=2.0 * L / delta)
    assert chk.strength_ok and not chk.deflection_ok and chk.status is Status.ALERT
    assert chk.issues[-1].startswith("ELS 1.0D + 1.0L")


# --------------------------------------------------------------------------- #
# Test 4: extremos deterministas (empates numéricos de |valor|)
# --------------------------------------------------------------------------- #


def test_extreme_tie_symmetric_beam_is_deterministic():
    """Viga simple simétrica (q_D uniforme + P_L en L/2): V(0) = +V_u/2 = −V(L), con
    V_u = 1.2·|q_D|·L + 1.6·|P_L| (equilibrio vertical). Empate de |V| ⇒ 1ª estación de la 1ª barra
    (x = 0, V > 0), igual en la superposición y en la corrida única, en cualquier plataforma."""
    q_D, P_L = -12.0, -35_000.0
    V_half = (1.2 * abs(q_D) * L + 1.6 * abs(P_L)) / 2  # 71 200 N
    combined = solve_combinations(beam(q_D=q_D, P_L=P_L), [COMB_1_2D_1_6L])[0]
    scaled = solve(build_beam(L, SIMPLE, SEC, MAT, [BeamPointLoad(L / 2, Fy=1.6 * P_L)],
                              [BeamDistLoad(0.0, L, 1.2 * q_D, 1.2 * q_D)]))
    for res in (combined, scaled):
        assert abs(res.members[0].V[0]) == pytest.approx(abs(res.members[-1].V[-1]), rel=1e-12)  # empate
        e = res.extreme("V")
        assert e.value == pytest.approx(V_half, rel=REL) and e.value > 0.0
        assert e.member == 0 and e.x == 0.0


def test_extreme_tie_rule_with_rounding_noise():
    """|V| que difieren en 1e-14 (ruido de redondeo) empatan ⇒ 1ª estación; una diferencia real
    (1e-9 relativo, por encima de la tolerancia 1e-12) gana aunque esté más adelante."""
    from dataclasses import replace

    base = solve(beam(q_D=-10.0))
    mr = base.members[0]
    for later, expected in ((-100.0 * (1 + 1e-14), 100.0), (-100.0 * (1 + 1e-9), -100.0 * (1 + 1e-9))):
        V = np.zeros_like(mr.V)
        V[0], V[-1] = 100.0, later
        res = Results(base.displacements, base.reactions, [replace(mr, V=V)])
        assert res.extreme("V").value == expected


# --------------------------------------------------------------------------- #
# Test 5: una sola resolución reutilizable (combine_cases / case_results)
# --------------------------------------------------------------------------- #


def test_combine_cases_and_check_with_case_results(monkeypatch):
    """check_combinations(case_results=solve_load_cases(m)) no vuelve a ensamblar K y da lo mismo."""
    from core.solver import combine_cases

    model = beam(q_D=-12.0, P_L=-35_000.0)
    cases = solve_load_cases(model)
    combos = [COMB_1_4D, COMB_1_2D_1_6L, COMB_SERVICE_D_L]
    for a, b in zip(combine_cases(cases, combos), solve_combinations(model, combos), strict=True):
        assert_same_results(a, b)

    calls = {"n": 0}
    original = solver_mod._assemble

    def spy(m):
        calls["n"] += 1
        return original(m)

    monkeypatch.setattr(solver_mod, "_assemble", spy)
    reused = check_combinations(model, case_results=cases)
    assert calls["n"] == 0
    fresh = check_combinations(model)
    assert calls["n"] == 1
    assert reused.governing is fresh.governing and reused.utilization == pytest.approx(fresh.utilization)
    assert len(reused.uls_results) == 2
    assert_same_results(reused.uls_results[1], reused.governing_results)  # 1.2D + 1.6L gobierna
    with pytest.raises(ValueError, match="hipótesis D"):
        combine_cases({}, combos)


def test_issue_texts_use_sy_notation():
    """Los mensajes usan S_y (fluencia), no Fy (que en el glosario es una fuerza)."""
    q_L = -((0.95 * SY * I / C * 8 / L**2) - 1.2 * 5.0) / 1.6
    chk = check_combinations(beam(q_D=-5.0, q_L=q_L), deflection_limit_ratio=0.0)
    assert "φ·Sy" in chk.issues[0] and "Fy" not in chk.issues[0]
    assert "Sy =" in check_combinations(beam(q_D=-5.0, q_L=2 * q_L), deflection_limit_ratio=0.0).issues[0]


# --------------------------------------------------------------------------- #
# Test 6: hipótesis de q y H en el pórtico
# --------------------------------------------------------------------------- #


def test_portal_frame_load_cases():
    """build_portal_frame: q y H en D por defecto; con beam_q_case / lateral_case = L pasan a L.

    Con q en L gobierna 1.2D + 1.6L (q_L ≫ D), y ΣRy de 1.2D + 1.6L = −1.6·q·luz (sin peso propio).
    """
    from core.builders import build_portal_frame

    span, height, qv, H = 8000.0, 4000.0, -12.0, 10_000.0
    default = build_portal_frame(span, height, SEC, SEC, MAT, ST.FIXED, ST.FIXED, beam_q=qv, lateral_load=H)
    assert {d.case for d in default.distributed_loads} == {LoadCase.DEAD}
    assert default.nodal_loads[0].case is LoadCase.DEAD
    live = build_portal_frame(span, height, SEC, SEC, MAT, ST.FIXED, ST.FIXED, beam_q=qv, lateral_load=H,
                              beam_q_case=LoadCase.LIVE, lateral_case=LoadCase.LIVE)
    assert {d.case for d in live.distributed_loads} == {LoadCase.LIVE}
    assert live.nodal_loads[0].case is LoadCase.LIVE
    chk = check_combinations(live)
    assert chk.governing is COMB_1_2D_1_6L
    Ry = sum(r[1] for r in chk.governing_results.reactions.values())
    assert Ry == pytest.approx(-1.6 * qv * span, rel=REL)
    Rx = sum(r[0] for r in chk.governing_results.reactions.values())
    assert Rx == pytest.approx(-1.6 * H, rel=REL)
