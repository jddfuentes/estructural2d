"""Punto de entrada Streamlit: `streamlit run app.py`.

Sólo orquesta: entradas (ui.inputs) -> core (solve + verificación) -> gráficos (ui.plots).
Nada de cálculo estructural acá.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from core.design import suggest_section
from core.materials import family_warnings
from core.model import COMB_1_2D_1_6L, COMB_1_4D, COMB_SERVICE_D_L, LoadCase, Model
from core.reports import build_pdf_report
from core.solver import Results, StructuralError, solve_combinations, solve_load_cases
from core.verification import CombinationCheck, SafetyCheck, Status, check_combinations
from ui.inputs import AppInputs, sidebar
from ui.plots import plot_diagram, plot_structure
from ui.theory import render_theory

st.set_page_config(page_title="Estructural 2D · Predimensionamiento", page_icon="📐", layout="wide")

STATUS_STYLE = {
    Status.OK: ("#15803D", "#DCFCE7", "VERIFICA"),
    Status.ALERT: ("#B45309", "#FEF3C7", "ALERTA"),
    Status.FAIL: ("#B91C1C", "#FEE2E2", "FLUENCIA"),
}


def _pdf_filename(title: str) -> str:
    normalized = title.strip().encode("ascii", "ignore").decode("ascii")
    filename = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return f"{filename or 'memoria'}.pdf"


def fs_badge(chk: SafetyCheck | CombinationCheck) -> None:
    if isinstance(chk, CombinationCheck):
        chk = chk.strength
    fg, bg, label = STATUS_STYLE[chk.status]
    fs_txt = "∞" if chk.fs == float("inf") else f"{chk.fs:.2f}"
    st.markdown(
        f"""<div style="background:{bg};border-left:6px solid {fg};padding:0.6rem 0.9rem;border-radius:6px">
        <div style="color:{fg};font-size:0.8rem;font-weight:700;letter-spacing:.06em">{label}</div>
        <div style="color:{fg};font-size:1.9rem;font-weight:800;line-height:1.1">FS = {fs_txt}</div>
        <div style="color:{fg};font-size:0.8rem">mín. {chk.fs_min:g} · Sy = {chk.sy:g} MPa</div></div>""",
        unsafe_allow_html=True,
    )


def reactions_table(model: Model, res: Results) -> pd.DataFrame:
    rows = []
    for sup in model.supports:
        Rx, Ry, Mz = res.reactions[sup.node]
        x, y = model.nodes[sup.node]
        rows.append({"Nodo": sup.node, "Tipo": sup.type.value, "x [m]": x / 1e3, "y [m]": y / 1e3,
                     "Rx [kN]": Rx / 1e3, "Ry [kN]": Ry / 1e3, "Mz [kN·m]": Mz / 1e6})
    return pd.DataFrame(rows)


def main() -> None:
    inp: AppInputs = sidebar()
    st.title("📐 Estructural 2D — predimensionamiento rápido")
    st.caption("Método directo de rigidez · Euler-Bernoulli · elástico lineal · unidades SI")

    model: Model | None = None
    try:
        model = inp.model()
        chk = check_combinations(model, deflection_limit_ratio=inp.deflection_limit_ratio)
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
        fs_badge(chk)
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
        fig_deformed = plot_structure(model, res, inp.deformed_scale)
        try:
            img_bytes = fig_deformed.to_image(format="png", width=900, height=350, scale=2)
        except Exception:
            st.sidebar.caption("Nota: Gráfico no incrustado en PDF (kaleido no disponible).")
            img_bytes = None
        pdf = build_pdf_report(model, res, chk, report_title, author,
                               reference_length=chk.service_check.reference_length, image_bytes=img_bytes)
        st.download_button("Descargar memoria", data=pdf, file_name=_pdf_filename(report_title),
                           mime="application/pdf")

    # ---- Gráficos ---------------------------------------------------------- #
    case_results = solve_load_cases(model)
    elu_results = solve_combinations(model, [COMB_1_4D, COMB_1_2D_1_6L])
    diagram_options = {
        f"ELU: {COMB_1_4D.name}": elu_results[0],
        f"ELU: {COMB_1_2D_1_6L.name}": elu_results[1],
        f"ELS: {COMB_SERVICE_D_L.name}": chk.service_results,
        "Hipótesis individual: D (Muerta)": case_results[LoadCase.DEAD],
    }
    if LoadCase.LIVE in case_results:
        diagram_options["Hipótesis individual: L (Viva)"] = case_results[LoadCase.LIVE]
    selected_diagram = st.selectbox("Solicitaciones a visualizar", list(diagram_options))
    diagram_results = diagram_options[selected_diagram]
    tabs = st.tabs(["Estructura y deformada", "Momento M", "Corte V", "Normal N", "Tensión σ", "Reacciones",
                    "Predimensionamiento", "Bases teóricas"])
    with tabs[0]:
        st.plotly_chart(fig_deformed, width="stretch")
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
        st.caption("Reacciones en ejes globales: Rx → +, Ry ↑ +, Mz antihorario +.")
    with tabs[6]:
        _design_tab(inp)
    with tabs[7]:
        render_theory()


def _design_tab(inp: AppInputs) -> None:
    if inp.family == "Personalizado":
        st.info("Elegí una familia del catálogo para buscar el perfil más liviano que verifica.")
        return
    target = "la viga" if inp.kind == "Viga" else "la viga del pórtico (columnas fijas)"
    _, c2 = st.columns(2)
    st.write(f"Perfil **{inp.family}** de menor peso para {target} con FS ≥ {inp.fs_min:g}"
             + (f" y δ ≤ L/{inp.deflection_limit_ratio:g}" if inp.deflection_limit_ratio else "") + ":")
    sug = suggest_section(inp.builder, inp.family, inp.fs_min,
                          deflection_limit_ratio=inp.deflection_limit_ratio)
    if sug is None:
        st.error("Ningún perfil de la familia verifica. Probá otra familia o reducí la luz/cargas.")
        return
    s = sug.section
    c2.metric(s.name, f"{s.mass_per_m:.1f} kg/m", delta=f"FS = {sug.check.fs:.2f}", delta_color="off")
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
