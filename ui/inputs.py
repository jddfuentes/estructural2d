"""Entradas de usuario (sidebar de Streamlit) -> modelo del core.

Toda conversión de unidades de ingeniería (m, kN, kN/m, kN·m) a unidades
del core (mm, N, N/mm, N·mm) y de signos "+ hacia abajo" a ejes globales
ocurre SOLO aquí. El core nunca ve unidades de UI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam, build_portal_frame
from core.materials import DEFAULT_MATERIAL, DEFAULT_SECTION, MATERIALS, SECTIONS, Material, Section, custom
from core.model import Model, SupportType
from core.verification import FS_MIN_DEFAULT

M = 1e3  # m -> mm
KN = 1e3  # kN -> N
KNM = 1e6  # kN·m -> N·mm

SUPPORT_LABELS = {t.value: t for t in SupportType}


@dataclass(frozen=True)
class AppInputs:
    kind: str  # "Viga" | "Pórtico"
    material: Material
    section: Section  # viga (o viga del pórtico)
    column_section: Section | None
    family: str
    fs_min: float
    moment_tension_side: bool
    deformed_scale: float | None
    builder: Callable[[Section], Model]  # rearma el modelo con otro perfil (predimensionamiento)
    span_mm: float

    def model(self) -> Model:
        return self.builder(self.section)


# --------------------------------------------------------------------------- #


def _section_picker(label: str, key: str) -> tuple[Section, str]:
    families = [*SECTIONS.keys(), "Personalizado"]
    fam = st.selectbox(f"{label}: familia", families, index=families.index(DEFAULT_SECTION[0]),
                       key=f"{key}_fam")
    if fam == "Personalizado":
        c1, c2 = st.columns(2)
        I_cm4 = c1.number_input("I [cm⁴]", min_value=0.01, value=1943.0, key=f"{key}_I")
        A_cm2 = c2.number_input("A [cm²]", min_value=0.01, value=28.5, key=f"{key}_A")
        h = st.number_input("Altura h [mm]", min_value=1.0, value=200.0, key=f"{key}_h")
        return custom("Personalizado", A=A_cm2 * 100, I=I_cm4 * 1e4, c=h / 2), fam
    names = list(SECTIONS[fam].keys())
    default = names.index(DEFAULT_SECTION[1]) if DEFAULT_SECTION[1] in names else len(names) // 2
    name = st.selectbox(f"{label}: perfil", names, index=default, key=f"{key}_name")
    sec = SECTIONS[fam][name]
    st.caption(f"A = {sec.A / 100:.2f} cm² · I = {sec.I / 1e4:,.0f} cm⁴ · W = {sec.W / 1e3:,.1f} cm³ · "
               f"{sec.mass_per_m:.1f} kg/m")
    return sec, fam


def _editor(df: pd.DataFrame, key: str, column_config: dict[str, Any]) -> pd.DataFrame:
    out = st.data_editor(df, key=key, num_rows="dynamic", hide_index=True, width="stretch",
                         column_config=column_config)
    return out.dropna(how="any")


def sidebar() -> AppInputs:
    sb = st.sidebar
    with sb:
        st.header("Modelo")
        kind = st.radio("Tipo de estructura", ["Viga", "Pórtico"], horizontal=True)

        st.subheader("Material y sección")
        mat_names = list(MATERIALS)
        material = MATERIALS[st.selectbox("Material", mat_names, index=mat_names.index(DEFAULT_MATERIAL))]
        st.caption(f"E = {material.E:,.0f} MPa · Sy = {material.Sy:g} MPa · Su = {material.Su:g} MPa")
        if kind == "Viga":
            section, family = _section_picker("Viga", "beam")
            column_section = None
        else:
            section, family = _section_picker("Viga", "beam")
            column_section, _ = _section_picker("Columnas", "col")
        self_weight = st.checkbox("Incluir peso propio", value=True)

        st.subheader("Geometría y cargas")
        st.caption("Cargas: **+ hacia abajo** · momentos **+ antihorario** · "
                   "horizontales **+ hacia la derecha**")
        if kind == "Viga":
            builder, span = _beam_inputs(material, self_weight)
        else:
            builder, span = _portal_inputs(material, column_section, self_weight)  # type: ignore[arg-type]

        st.subheader("Verificación y gráficos")
        fs_min = st.number_input("FS mínimo admisible", min_value=1.0, max_value=5.0,
                                 value=FS_MIN_DEFAULT, step=0.1)
        tension_side = st.toggle("Momento del lado traccionado", value=True)
        auto = st.toggle("Escala de deformada automática", value=True)
        scale = None if auto else st.number_input("Factor de escala", min_value=1.0, value=100.0, step=10.0)

    return AppInputs(kind=kind, material=material, section=section, column_section=column_section,
                     family=family, fs_min=fs_min, moment_tension_side=tension_side,
                     deformed_scale=scale, builder=builder, span_mm=span)


def _beam_inputs(material: Material, self_weight: bool) -> tuple[Callable[[Section], Model], float]:
    L = st.number_input("Longitud L [m]", min_value=0.1, value=6.0, step=0.5)

    st.markdown("**Apoyos**")
    sup_df = _editor(
        pd.DataFrame({"x [m]": [0.0, L], "Tipo": [SupportType.PINNED.value, SupportType.ROLLER.value]}),
        "supports",
        {"x [m]": st.column_config.NumberColumn(min_value=0.0, max_value=L, step=0.1, format="%.3f"),
         "Tipo": st.column_config.SelectboxColumn(options=list(SUPPORT_LABELS), required=True)},
    )
    st.markdown("**Cargas puntuales**")
    pl_df = _editor(
        pd.DataFrame({"x [m]": [L / 2], "P [kN] ↓": [20.0], "M [kN·m] ↺": [0.0]}),
        "point_loads",
        {"x [m]": st.column_config.NumberColumn(min_value=0.0, max_value=L, step=0.1, format="%.3f")},
    )
    st.markdown("**Cargas distribuidas**")
    dl_df = _editor(
        pd.DataFrame({"x ini [m]": [0.0], "x fin [m]": [L],
                      "q ini [kN/m] ↓": [10.0], "q fin [kN/m] ↓": [10.0]}),
        "dist_loads",
        {"x ini [m]": st.column_config.NumberColumn(min_value=0.0, max_value=L, format="%.3f"),
         "x fin [m]": st.column_config.NumberColumn(min_value=0.0, max_value=L, format="%.3f")},
    )

    supports = [BeamSupport(float(r["x [m]"]) * M, SUPPORT_LABELS[str(r["Tipo"])])
                for _, r in sup_df.iterrows()]
    point_loads = [
        BeamPointLoad(float(r["x [m]"]) * M, Fy=-float(r["P [kN] ↓"]) * KN, Mz=float(r["M [kN·m] ↺"]) * KNM)
        for _, r in pl_df.iterrows()
    ]
    dist_loads = [
        BeamDistLoad(float(r["x ini [m]"]) * M, float(r["x fin [m]"]) * M,
                     -float(r["q ini [kN/m] ↓"]), -float(r["q fin [kN/m] ↓"]))  # kN/m = N/mm
        for _, r in dl_df.iterrows()
    ]

    def builder(sec: Section) -> Model:
        return build_beam(L * M, supports, sec, material, point_loads, dist_loads, self_weight)

    return builder, L * M


def _portal_inputs(material: Material, column: Section, self_weight: bool
                   ) -> tuple[Callable[[Section], Model], float]:
    c1, c2 = st.columns(2)
    span = c1.number_input("Luz [m]", min_value=0.5, value=8.0, step=0.5)
    height = c2.number_input("Altura [m]", min_value=0.5, value=4.0, step=0.5)
    opts = [SupportType.FIXED.value, SupportType.PINNED.value]
    b1, b2 = st.columns(2)
    base_l = SUPPORT_LABELS[b1.selectbox("Base izquierda", opts)]
    base_r = SUPPORT_LABELS[b2.selectbox("Base derecha", opts)]
    q = st.number_input("q sobre la viga [kN/m] ↓", value=12.0, step=1.0)
    H = st.number_input("Carga lateral en nudo sup. izq. [kN] →", value=10.0, step=1.0)
    st.markdown("**Cargas puntuales sobre la viga**")
    pl_df = _editor(
        pd.DataFrame({"x [m]": [span / 2], "P [kN] ↓": [0.0]}),
        "portal_point_loads",
        {"x [m]": st.column_config.NumberColumn(min_value=0.0, max_value=span, step=0.1, format="%.3f")},
    )
    point_loads = [BeamPointLoad(float(r["x [m]"]) * M, Fy=-float(r["P [kN] ↓"]) * KN)
                   for _, r in pl_df.iterrows() if float(r["P [kN] ↓"]) != 0.0]

    def builder(sec: Section) -> Model:
        return build_portal_frame(span * M, height * M, column, sec, material, base_l, base_r,
                                  beam_q=-q, lateral_load=H * KN, beam_point_loads=point_loads,
                                  self_weight=self_weight)

    return builder, span * M
