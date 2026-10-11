"""Punto de entrada Streamlit: `streamlit run app.py`.

Sólo orquesta: entradas (ui.inputs) -> core (solve + verificación) -> gráficos (ui.plots).
Nada de cálculo estructural acá.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.design import Suggestion, suggest_section
from core.materials import Section, family_warnings
from core.model import COMB_SERVICE_D_L, LoadCase, Model, SupportType
from core.reports import build_pdf_report
from core.solver import Results, StructuralError, solve_load_cases
from core.verification import FS_MIN_DEFAULT, CombinationCheck, Status, check_combinations
from ui.inputs import AppInputs, sidebar
from ui.plots import plot_diagram, plot_structure
from ui.theory import render_theory

st.set_page_config(page_title="Estructural 2D · Predimensionamiento", page_icon="📐", layout="wide")

# Severidad del veredicto -> (texto, fondo). El rótulo sale de `CombinationCheck.verdict`.
STATUS_STYLE = {
    Status.OK: ("#15803D", "#DCFCE7"),
    Status.ALERT: ("#B45309", "#FEF3C7"),
    Status.FAIL: ("#B91C1C", "#FEE2E2"),
}


def _pdf_filename(title: str) -> str:
    normalized = title.strip().encode("ascii", "ignore").decode("ascii")
    filename = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return f"{filename or 'memoria'}.pdf"


def verdict_badge(chk: CombinationCheck) -> None:
    """Estado GLOBAL (ELU + ELS): rojo si no verifica ELU, amarillo si sólo falla la flecha de servicio."""
    severity, label = chk.verdict
    fg, bg = STATUS_STYLE[severity]
    sv = chk.service_check
    ratio = "∞" if not math.isfinite(sv.deflection_ratio) else f"{sv.deflection_ratio:,.0f}"
    limit = f" (adm. L/{sv.deflection_limit_ratio:g})" if sv.deflection_limit_ratio > 0 else ""
    util = f"ELU {chk.utilization:.0%}"
    st.markdown(
        f"""<div style="background:{bg};border-left:6px solid {fg};padding:0.6rem 0.9rem;border-radius:6px">
        <div style="color:{fg};font-size:0.8rem;font-weight:700;letter-spacing:.06em">{label}</div>
        <div style="color:{fg};font-size:1.9rem;font-weight:800;line-height:1.1">{util}</div>
        <div style="color:{fg};font-size:0.8rem">σu/(φ·Sy) · {chk.governing.name} · φ = {chk.phi:g} ·
        Sy = {chk.strength.sy:g} MPa<br>ELS {chk.service_combination.name}: L/{ratio}{limit}</div></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Caché por modelo (AGENTS.md §6: el core no cachea; la UI sí). La clave es `repr(model)`, que
# cubre geometría, secciones, cargas, apoyos y resortes: cambiar de pestaña o de diagrama no vuelve
# a resolver, ni a exportar el PNG, ni a generar el PDF.
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, max_entries=16)
def _analysis(key: str, _model: Model, deflection_limit_ratio: float
              ) -> tuple[dict[LoadCase, Results], CombinationCheck]:
    """D y L se resuelven UNA vez (K ensamblada una vez); las combinaciones se superponen."""
    cases = solve_load_cases(_model)
    return cases, check_combinations(_model, deflection_limit_ratio=deflection_limit_ratio,
                                     case_results=cases)


@st.cache_data(show_spinner=False, max_entries=16)
def _png(key: str, _fig: go.Figure) -> bytes | None:
    """PNG de la figura para el PDF (kaleido); None si kaleido/Chrome no está disponible."""
    try:
        return bytes(_fig.to_image(format="png", width=900, height=350, scale=2))
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=16)
def _pdf(key: str, _model: Model, _chk: CombinationCheck, title: str, author: str,
         image: bytes | None) -> bytes:
    return build_pdf_report(_model, _chk.governing_results, _chk, title, author,
                            reference_length=_chk.service_check.reference_length, image_bytes=image)


@st.cache_data(show_spinner=False, max_entries=16)
def _suggestion(key: str, _build: Callable[[Section], Model], family: str,
                deflection_limit_ratio: float, phi: float) -> Suggestion | None:
    return suggest_section(_build, family, FS_MIN_DEFAULT,  # fs_min no se usa con φ
                           deflection_limit_ratio=deflection_limit_ratio, phi=phi)


def reactions_table(model: Model, res: Results) -> pd.DataFrame:
    """Reacciones de apoyos rígidos y de resortes (incluye nodos sólo con resorte)."""
    supports = {sup.node: sup.type for sup in model.supports}
    rows = []
    springs_by_node = {sp.node: sp for sp in model.springs}
    for node, (Rx, Ry, Mz) in res.reactions.items():
        kind = supports.get(node)
        tipo = kind.value if kind is not None and kind is not SupportType.FREE else ""
        sp = springs_by_node.get(node)
        if sp is not None and not sp.active(kind).is_null:  # sólo resortes que trabajan
            tipo = f"{tipo} + elástico" if tipo else "elástico"
        x, y = model.nodes[node]
        rows.append({"Nodo": node, "Tipo": tipo or SupportType.FREE.value, "x [m]": x / 1e3,
                     "y [m]": y / 1e3, "Rx [kN]": Rx / 1e3, "Ry [kN]": Ry / 1e3, "Mz [kN·m]": Mz / 1e6})
    return pd.DataFrame(rows)


def main() -> None:
    inp: AppInputs = sidebar()
    st.title("📐 Estructural 2D — predimensionamiento rápido")
    st.caption("Método directo de rigidez · Euler-Bernoulli · elástico lineal · unidades SI")

    model: Model | None = None
    try:
        model = inp.model()
        key = repr(model)
        case_results, chk = _analysis(key, model, inp.deflection_limit_ratio)
    except (ValueError, StructuralError) as exc:
        st.error(f"**No se pudo resolver el modelo:** {exc}")
        if model is not None:
            st.plotly_chart(plot_structure(model), width="stretch")
        return

    res = chk.governing_results
    M, V, N = res.extreme("M"), res.extreme("V"), res.extreme("N")

    # ---- Indicadores ------------------------------------------------------- #
    c0, c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1, 1])
    with c0:
        verdict_badge(chk)
    strength = chk.strength
    c1.metric("σ máx", f"{strength.sigma_max:.1f} MPa", help="|N/A ± M·c/I| máximo en ELU")
    c2.metric("M máx", f"{abs(M.value) / 1e6:.2f} kN·m",
              help=f"Barra {M.member}, x global = {M.point[0] / 1e3:.2f} m")
    c3.metric("V máx", f"{abs(V.value) / 1e3:.2f} kN")
    ratio = chk.service_check.deflection_ratio
    limit_help = (
        "sin límite de flecha"
        if inp.deflection_limit_ratio == 0.0
        else f"límite de servicio L/{inp.deflection_limit_ratio:g}"
    )
    c4.metric("δ máx", f"{chk.delta_max:.2f} mm", help=f"Flecha ELS; {limit_help}",
              delta=f"L/{ratio:,.0f}" if ratio != float("inf") else None, delta_color="off")

    for issue in chk.issues:
        st.warning(f"Servicio: {issue[4:]}" if issue.startswith("ELS ") else issue)

    # ---- Memoria ---------------------------------------------------------- #
    with st.expander("Memoria de cálculo (PDF)"):
        report_title = st.text_input("Título", value="Memoria de Cálculo")
        author = st.text_input("Autor", value="")
        # Deformada de servicio (1.0D + 1.0L): la misma flecha que el indicador δ y que §3.3 del PDF
        fig_deformed = plot_structure(model, chk.service_results, inp.deformed_scale)
        fig_key = f"{key}|{inp.deflection_limit_ratio}|{inp.deformed_scale}"
        img_bytes = _png(fig_key, fig_deformed)
        if img_bytes is None:
            st.caption("Nota: gráfico no incrustado en el PDF (kaleido no disponible).")
        pdf = _pdf(fig_key, model, chk, report_title, author, img_bytes)
        st.download_button("Descargar memoria", data=pdf, file_name=_pdf_filename(report_title),
                           mime="application/pdf")

    # ---- Gráficos ---------------------------------------------------------- #
    # Sin re-resolver: combinaciones e hipótesis salen del mismo análisis (K ensamblada una vez)
    diagram_options: dict[str, Results] = {}
    for (comb, _), res_c in zip(chk.strength_checks, chk.uls_results, strict=True):
        gov = " — gobierna" if comb is chk.governing else ""
        diagram_options[f"ELU: {comb.name}{gov}"] = res_c
    diagram_options[f"ELS: {chk.service_combination.name}"] = chk.service_results
    diagram_options["Hipótesis D (permanente)"] = case_results[LoadCase.DEAD]
    if LoadCase.LIVE in case_results:
        diagram_options["Hipótesis L (sobrecarga)"] = case_results[LoadCase.LIVE]
    labels = list(diagram_options)
    governing_label = next(lbl for lbl in labels if lbl.endswith("— gobierna"))
    selected_diagram = st.selectbox("Solicitaciones a visualizar", labels,
                                    index=labels.index(governing_label))
    diagram_results = diagram_options[selected_diagram]
    tabs = st.tabs(["Estructura y deformada", "Momento M", "Corte V", "Normal N", "Tensión σ", "Reacciones",
                    "Predimensionamiento", "Bases teóricas"])
    with tabs[0]:
        st.plotly_chart(fig_deformed, width="stretch")
        st.caption(f"Deformada de servicio ({COMB_SERVICE_D_L.name}, cargas sin mayorar).")
    with tabs[1]:
        st.plotly_chart(plot_diagram(model, diagram_results, "M", inp.moment_tension_side), width="stretch")
    with tabs[2]:
        st.plotly_chart(plot_diagram(model, diagram_results, "V"), width="stretch")
    with tabs[3]:
        if abs(N.value) < 1e-6:
            st.info("Esfuerzo normal nulo en toda la estructura.")
        st.plotly_chart(plot_diagram(model, diagram_results, "N"), width="stretch")
    with tabs[4]:
        st.plotly_chart(plot_diagram(model, diagram_results, "sigma"), width="stretch")
        st.caption("σ = N/A ± M·c/I en fibra extrema. "
                   "No incluye pandeo, PLT, corte ni concentración de tensiones.")
    with tabs[5]:
        st.dataframe(reactions_table(model, res), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("x [m]", "y [m]", "Rx [kN]", "Ry [kN]", "Mz [kN·m]")})
        st.caption(f"Reacciones de la combinación ELU gobernante ({chk.governing.name}, cargas mayoradas), "
                   "en ejes globales: Rx → +, Ry ↑ +, Mz antihorario +.")
    with tabs[6]:
        _design_tab(inp, chk)
    with tabs[7]:
        render_theory()


def _design_tab(inp: AppInputs, chk: CombinationCheck) -> None:
    if inp.family == "Personalizado":
        st.info("Elegí una familia del catálogo para buscar el perfil más liviano que verifica.")
        return
    target = "la viga" if inp.kind == "Viga" else "la viga del pórtico (columnas fijas)"
    _, c2 = st.columns(2)
    uls = " y ".join(comb.name for comb, _ in chk.strength_checks)
    st.write(f"Perfil **{inp.family}** de menor peso para {target} con σu ≤ φ·Sy (φ = {chk.phi:g}; {uls})"
             + (f" y δ ≤ L/{inp.deflection_limit_ratio:g} con {chk.service_combination.name}"
                if inp.deflection_limit_ratio else "") + ":")
    # Mismo criterio que el veredicto superior: verificación completa por combinaciones
    sug = _suggestion(f"{repr(inp.model())}|{inp.family}", inp.builder, inp.family,
                      inp.deflection_limit_ratio, chk.phi)
    if sug is None:
        st.error("Ningún perfil de la familia verifica. Probá otra familia o reducí la luz/cargas.")
        return
    s = sug.section
    util = sug.combination.utilization if sug.combination is not None else sug.check.utilization
    c2.metric(s.name, f"{s.linear_mass(inp.material.rho):.1f} kg/m", delta=f"ELU {util:.0%}",
              delta_color="off")
    for warning in family_warnings(s.family):
        st.warning(warning)
    if s.name == inp.section.name:
        st.success("El perfil seleccionado ya es el más liviano que verifica.")
    elif s.A < inp.section.A:
        saving = 1 - s.A / inp.section.A
        st.info(f"Podés bajar de {inp.section.name} a **{s.name}** ({saving:.0%} menos peso).")
    else:
        st.warning(f"{inp.section.name} no alcanza: subir a **{s.name}**.")


main()
