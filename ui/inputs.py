"""Entradas de usuario (sidebar de Streamlit) -> modelo del core.

Toda conversión de unidades de ingeniería (m, kN, kN/m, kN·m) a unidades
del core (mm, N, N/mm, N·mm) y de signos "+ hacia abajo" a ejes globales
ocurre SOLO aquí. El core nunca ve unidades de UI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
import streamlit as st

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam, build_portal_frame
from core.materials import (
    DEFAULT_MATERIAL,
    DEFAULT_SECTION,
    FAMILY_NOTES,
    MATERIALS,
    SECTIONS,
    Material,
    Section,
    custom,
    family_warnings,
)
from core.model import Model, Spring, SupportType
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
    deflection_limit_ratio: float = 300.0

    def model(self) -> Model:
        return self.builder(self.section)


# --------------------------------------------------------------------------- #


def _section_picker(label: str, key: str) -> tuple[Section, str]:
    families = [*SECTIONS.keys(), "Personalizado"]
    fam = st.selectbox(f"{label}: familia", families, index=families.index(DEFAULT_SECTION[0]),
                       key=f"{key}_fam")
    note = FAMILY_NOTES.get(fam)
    if note:
        st.caption(note)
    for warning in family_warnings(fam):
        st.warning(warning)
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


def _member_release_editor(labels: list[str], key: str) -> pd.DataFrame:
    """Editor compacto de rótulas por barra; los índices son los del modelo generado."""
    return _editor(
        pd.DataFrame({
            "Barra": labels,
            "Rótula inicio": [False] * len(labels),
            "Rótula fin": [False] * len(labels),
        }),
        key,
        {
            "Barra": st.column_config.TextColumn(disabled=True),
            "Rótula inicio": st.column_config.CheckboxColumn(default=False),
            "Rótula fin": st.column_config.CheckboxColumn(default=False),
        },
    )


def _apply_member_releases(model: Model, releases: pd.DataFrame) -> Model:
    """Propaga los booleanos del editor a las barras del modelo construido."""
    model.members = [
        replace(
            mem,
            release_start=bool(releases.iloc[i]["Rótula inicio"]),
            release_end=bool(releases.iloc[i]["Rótula fin"]),
        )
        for i, mem in enumerate(model.members)
    ]
    return model


def _spring_if_present(node: int, kx: float, ky: float, krz: float) -> Spring | None:
    """Construye un resorte sólo cuando alguna rigidez fue asignada en la UI."""
    if kx == 0.0 and ky == 0.0 and krz == 0.0:
        return None
    return Spring(node, kx=kx, ky=ky, krz=krz)


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
        deflection_limit_ratio = st.number_input(
            "Límite de flecha L/…", min_value=0.0, value=300.0, step=50.0,
            help="0 = sin límite de flecha",
        )
        tension_side = st.toggle("Momento del lado traccionado", value=True)
        auto = st.toggle("Escala de deformada automática", value=True)
        scale = None if auto else st.number_input("Factor de escala", min_value=1.0, value=100.0, step=10.0)

    return AppInputs(kind=kind, material=material, section=section, column_section=column_section,
                     family=family, fs_min=fs_min, moment_tension_side=tension_side,
                     deformed_scale=scale, builder=builder, span_mm=span,
                     deflection_limit_ratio=deflection_limit_ratio)


def _beam_inputs(material: Material, self_weight: bool) -> tuple[Callable[[Section], Model], float]:
    L = st.number_input("Longitud L [m]", min_value=0.1, value=6.0, step=0.5)

    st.markdown("**Apoyos**")
    sup_df = _editor(
        pd.DataFrame({
            "x [m]": [0.0, L],
            "Tipo": [SupportType.PINNED.value, SupportType.ROLLER.value],
            "Kx [kN/m]": [0.0, 0.0],
            "Ky [kN/m]": [0.0, 0.0],
            "Krz [kN·m/rad]": [0.0, 0.0],
        }),
        "supports",
        {
            "x [m]": st.column_config.NumberColumn(min_value=0.0, max_value=L, step=0.1, format="%.3f"),
            "Tipo": st.column_config.SelectboxColumn(options=list(SUPPORT_LABELS), required=True),
            "Kx [kN/m]": st.column_config.NumberColumn(min_value=0.0, step=100.0, format="%.1f"),
            "Ky [kN/m]": st.column_config.NumberColumn(min_value=0.0, step=100.0, format="%.1f"),
            "Krz [kN·m/rad]": st.column_config.NumberColumn(min_value=0.0, step=10.0, format="%.1f"),
        },
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
    xs = sorted({
        0.0, L,
        *[float(r["x [m]"]) for _, r in sup_df.iterrows()],
        *[float(r["x [m]"]) for _, r in pl_df.iterrows()],
        *[float(r["x ini [m]"]) for _, r in dl_df.iterrows()],
        *[float(r["x fin [m]"]) for _, r in dl_df.iterrows()],
    })
    release_df = _member_release_editor(
        [f"{i} ({a:.3f}–{b:.3f} m)" for i, (a, b) in enumerate(zip(xs, xs[1:], strict=False))],
        "beam_member_releases",
    )

    supports = [
        BeamSupport(float(r["x [m]"]) * M, SUPPORT_LABELS[str(r["Tipo"])])
        for _, r in sup_df.iterrows()
    ]
    spring_inputs = [
        (
            float(r["x [m]"]) * M,
            float(r["Kx [kN/m]"]),
            float(r["Ky [kN/m]"]),
            float(r["Krz [kN·m/rad]"]),
        )
        for _, r in sup_df.iterrows()
    ]
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
        model = build_beam(L * M, supports, sec, material, point_loads, dist_loads, self_weight)
        for x, kx, ky, krz in spring_inputs:
            node = min(range(len(model.nodes)), key=lambda i: abs(model.nodes[i][0] - x))
            spring = _spring_if_present(node, kx, ky, krz * KNM)
            if spring is not None:
                model.springs.append(spring)
        return _apply_member_releases(model, release_df)

    return builder, L * M


def _portal_inputs(material: Material, column: Section, self_weight: bool
                   ) -> tuple[Callable[[Section], Model], float]:
    c1, c2 = st.columns(2)
    span = c1.number_input("Luz [m]", min_value=0.5, value=8.0, step=0.5)
    height = c2.number_input("Altura [m]", min_value=0.5, value=4.0, step=0.5)
    opts = [SupportType.FIXED.value, SupportType.PINNED.value, SupportType.FREE.value]
    b1, b2 = st.columns(2)
    base_l = SUPPORT_LABELS[b1.selectbox("Base izquierda", opts)]
    base_r = SUPPORT_LABELS[b2.selectbox("Base derecha", opts)]
    st.markdown("**Resortes nodales en las bases**")
    kx_l, ky_l, krz_l = st.columns(3)
    kx_r, ky_r, krz_r = st.columns(3)
    left_spring = (
        kx_l.number_input("Kx izq. [kN/m]", min_value=0.0, step=100.0, key="portal_kx_l"),
        ky_l.number_input("Ky izq. [kN/m]", min_value=0.0, step=100.0, key="portal_ky_l"),
        krz_l.number_input("Krz izq. [kN·m/rad]", min_value=0.0, step=10.0, key="portal_krz_l"),
    )
    right_spring = (
        kx_r.number_input("Kx der. [kN/m]", min_value=0.0, step=100.0, key="portal_kx_r"),
        ky_r.number_input("Ky der. [kN/m]", min_value=0.0, step=100.0, key="portal_ky_r"),
        krz_r.number_input("Krz der. [kN·m/rad]", min_value=0.0, step=10.0, key="portal_krz_r"),
    )
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
    release_df = _member_release_editor(
        ["0 (col. izq.)", "1 (viga)", "2 (col. der.)"],
        "portal_member_releases",
    )

    def builder(sec: Section) -> Model:
        model = build_portal_frame(span * M, height * M, column, sec, material, base_l, base_r,
                                   beam_q=-q, lateral_load=H * KN, beam_point_loads=point_loads,
                                   self_weight=self_weight)
        for node, values in ((0, left_spring), (3, right_spring)):
            spring = _spring_if_present(node, values[0], values[1], values[2] * KNM)
            if spring is not None:
                model.springs.append(spring)
        return _apply_member_releases(model, release_df)

    return builder, span * M
