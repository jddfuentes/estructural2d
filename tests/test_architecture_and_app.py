"""Reglas de arquitectura (AGENTS.md §3) y prueba de humo de la app Streamlit."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "core": {"streamlit", "plotly", "pandas", "ui"},
    "ui/plots.py": {"streamlit"},
}
# ReportLab sólo en la capa de presentación del core: el solver no debe depender de él.
REPORTLAB_ALLOWED = {"core/reports.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("target,forbidden", FORBIDDEN.items())
def test_dependency_rule(target, forbidden):
    p = ROOT / target
    files = [p] if p.is_file() else sorted(p.rglob("*.py"))
    for f in files:
        bad = _imports(f) & forbidden
        assert not bad, f"{f.relative_to(ROOT)} importa {bad} (prohibido por AGENTS.md §3)"


def _full_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_reportlab_only_in_reports():
    """El solver no depende de ReportLab: sólo core/reports.py lo importa y nadie del core importa reports."""
    for f in sorted((ROOT / "core").rglob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        if rel in REPORTLAB_ALLOWED:
            continue
        mods = _full_imports(f)
        assert not any(m.split(".")[0] == "reportlab" for m in mods), f"{rel} importa reportlab"
        assert "core.reports" not in mods, f"{rel} importa core.reports"


@pytest.mark.parametrize("kind", ["Viga", "Pórtico"])
def test_app_smoke(kind):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if kind != "Viga":
        at.sidebar.radio[0].set_value(kind).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert "σ máx" in labels and "δ máx" in labels
    assert "Descargar memoria" in [button.label for button in at.download_button]


@pytest.mark.parametrize("kind", ["Viga", "Pórtico"])
@pytest.mark.parametrize("family", ["UPN", "Perfil W", "Perfil C (Conformado)"])
def test_app_new_profile_families(kind, family):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if kind != "Viga":
        at.sidebar.radio[0].set_value(kind).run()
    at.sidebar.selectbox(key="beam_fam").set_value(family).run()

    assert not at.exception, [e.message for e in at.exception]
    assert any(metric.label == "σ máx" for metric in at.metric)


@pytest.mark.parametrize("kind", ["Viga", "Pórtico"])
@pytest.mark.parametrize("family,has_warning", [("UPN", True), ("IPE", False)])
def test_app_profile_family_warning(kind, family, has_warning):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from core.materials import family_warnings

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if kind != "Viga":
        at.sidebar.radio[0].set_value(kind).run()
    at.sidebar.selectbox(key="beam_fam").set_value(family).run()

    assert not at.exception, [e.message for e in at.exception]
    rendered_warnings = [warning.value for warning in at.warning]
    family_messages = family_warnings(family)
    if has_warning:
        assert any(message in rendered_warnings for message in family_messages)
    else:
        assert not any(message in rendered_warnings for message in family_messages)


def test_app_deflection_warning_for_ipe_240_beam():
    """AppTest: el aviso de servicio se muestra separado del de resistencia."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.selectbox(key="beam_name").set_value("IPE 240")
    at.sidebar.number_input[0].set_value(8.0)
    at.sidebar.checkbox[0].set_value(False)
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    warning_text = [warning.value for warning in at.warning]
    assert any(text.startswith("Servicio") for text in warning_text)
    assert not any(text.startswith("Resistencia") for text in warning_text)


def test_member_releases_table_must_match_members():
    """O1: la tabla de rótulas tiene una fila por barra; si no coincide, ValueError (no IndexError)."""
    pytest.importorskip("streamlit")
    import pandas as pd

    from core.builders import BeamSupport, build_beam
    from core.materials import MATERIALS, get_section
    from core.model import SupportType
    from ui.inputs import _apply_member_releases

    m = build_beam(6000.0, [BeamSupport(0.0, SupportType.PINNED), BeamSupport(3000.0, SupportType.ROLLER),
                            BeamSupport(6000.0, SupportType.ROLLER)],
                   get_section("IPE 200"), MATERIALS["ASTM A36"])
    assert len(m.members) == 2
    one_row = pd.DataFrame({"Barra": ["0"], "Rótula inicio": [False], "Rótula fin": [True]})
    with pytest.raises(ValueError, match="una fila por barra"):
        _apply_member_releases(m, one_row)
    two_rows = pd.DataFrame({"Barra": ["0", "1"], "Rótula inicio": [False, True],
                             "Rótula fin": [True, False]})
    out = _apply_member_releases(m, two_rows)
    assert out.members[0].release_end and out.members[1].release_start


def test_app_badge_reflects_global_status_service_alert():
    """B3: si ELU verifica y la flecha de servicio no, el badge superior dice ALERTA (no VERIFICA).

    Modelo por defecto de la app (P = 20 kN en L/2 y q = 10 kN/m, ambas D) con L = 10 m y sin
    peso propio; el perfil IPE se elige con el core para que ELU verifique y ELS no.
    """
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam
    from core.materials import DEFAULT_MATERIAL, MATERIALS, SECTIONS
    from core.model import SupportType
    from core.verification import check_combinations

    span = 10_000.0

    def build(sec):
        return build_beam(span, [BeamSupport(0.0, SupportType.PINNED), BeamSupport(span, SupportType.ROLLER)],
                          sec, MATERIALS[DEFAULT_MATERIAL], [BeamPointLoad(span / 2, Fy=-20_000.0)],
                          [BeamDistLoad(0.0, span, -10.0, -10.0)], self_weight=False)

    name = next(s.name for s in sorted(SECTIONS["IPE"].values(), key=lambda s: s.A)
                if (c := check_combinations(build(s))).strength_ok and not c.deflection_ok)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.selectbox(key="beam_name").set_value(name)
    at.sidebar.number_input[0].set_value(span / 1000.0)
    at.sidebar.checkbox[0].set_value(False)
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    badge = next(md.value for md in at.markdown if "border-left:6px" in md.value)
    assert ">ALERTA<" in badge and ">VERIFICA<" not in badge
    assert any(w.value.startswith("Servicio") for w in at.warning)


def test_member_labels_follow_generated_model():
    """O1/O7: la tabla de rótulas tiene una fila por barra REAL; en el pórtico, una carga puntual
    parte la viga en dos (4 barras) y la base derecha es el último nodo (no el nodo 3)."""
    pytest.importorskip("streamlit")
    from core.builders import BeamPointLoad, build_portal_frame
    from core.materials import MATERIALS, get_section
    from core.model import SupportType
    from ui.inputs import _member_labels

    m = build_portal_frame(8000.0, 4000.0, get_section("IPE 240"), get_section("IPE 300"),
                           MATERIALS["ASTM A36"], SupportType.FIXED, SupportType.FIXED,
                           beam_point_loads=[BeamPointLoad(3000.0, Fy=-10_000.0)])
    labels = _member_labels(m)
    assert len(labels) == len(m.members) == 4
    assert labels[0] == "0 (col. izq.)" and labels[-1] == "3 (col. der.)"
    assert labels[1] == "1 (viga 0.000–3.000 m)" and labels[2] == "2 (viga 3.000–8.000 m)"
    assert m.nodes[-1] == (8000.0, 0.0) and len(m.nodes) == 5


def _portal_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Pórtico").run()
    return at


def test_app_portal_live_loads_and_elastic_base():
    """O7/O12: pórtico con q y H en hipótesis L y base izquierda libre sobre resortes (Kx, Ky, Krz):
    resuelve sin excepción, el diagrama por defecto es el ELU gobernante y existe la vista de L."""
    pytest.importorskip("streamlit")
    at = _portal_app()
    next(s for s in at.sidebar.selectbox if s.label == "Base izquierda").set_value("libre")
    at.sidebar.number_input(key="portal_kx_l").set_value(50_000.0)
    at.sidebar.number_input(key="portal_ky_l").set_value(100_000.0)
    at.sidebar.number_input(key="portal_krz_l").set_value(20_000.0)
    at.sidebar.selectbox(key="portal_q_case").set_value("L")
    at.sidebar.selectbox(key="portal_H_case").set_value("L")
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    diagram = next(s for s in at.selectbox if s.label == "Solicitaciones a visualizar")
    assert diagram.value.endswith("— gobierna") and "1.2D + 1.6L" in diagram.value
    assert "Hipótesis L (sobrecarga)" in diagram.options
    assert any(md.value.count("border-left:6px") for md in at.markdown)  # badge global


def test_app_warns_inert_spring_on_restrained_dof():
    """O2: Kx en una base articulada (ux restringido) no trabaja: aviso y sin resorte en el modelo."""
    pytest.importorskip("streamlit")
    at = _portal_app()
    next(s for s in at.sidebar.selectbox if s.label == "Base izquierda").set_value("articulado")
    at.sidebar.number_input(key="portal_kx_l").set_value(50_000.0)
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    assert any("Kx no trabaja" in w.value for w in at.sidebar.warning)


def test_app_portal_free_bases_with_springs_reactions():
    """AppTest: pórtico con bases "libre" + resortes resuelve y la tabla de reacciones las lista."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Pórtico").run()
    for label in ("Base izquierda", "Base derecha"):
        next(s for s in at.sidebar.selectbox if s.label == label).set_value("libre")
    for side in ("l", "r"):
        at.sidebar.number_input(key=f"portal_kx_{side}").set_value(20_000.0)
        at.sidebar.number_input(key=f"portal_ky_{side}").set_value(500_000.0)
        at.sidebar.number_input(key=f"portal_krz_{side}").set_value(30_000.0)
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    tables = [d.value for d in at.dataframe if "Rx [kN]" in d.value.columns]
    assert tables and set(tables[0]["Tipo"]) == {"elástico"}
    # La tabla es de la combinación ELU gobernante: sin cargas L gobierna 1.4D ⇒ ΣRx = −1.4·H = −14 kN
    assert abs(tables[0]["Rx [kN]"].sum() + 1.4 * 10.0) < 1e-6


# --------------------------------------------------------------------------- #
# Auditoría pre-release v0.1.0: B1 (eje débil en la sugerencia), B3 (caché del PDF), B5 (pórtico genérico)
# --------------------------------------------------------------------------- #


def _patched_app_script(root, overrides):
    """Corre app.py con datos iniciales propios en algunos `st.data_editor` (por clave) y guarda en
    `st.session_state["_downloads"][rótulo]` los bytes de cada `st.download_button`.

    AppTest no puede editar un `data_editor` ni leer el archivo de un `download_button`: se envuelven
    ambas funciones sólo durante esta corrida (se restauran en `finally`, así no afectan a otros AppTest
    del mismo proceso).
    """
    import runpy
    import sys

    import pandas as pd
    import streamlit as st

    if root not in sys.path:
        sys.path.insert(0, root)
    original, original_download = st.data_editor, st.download_button

    def data_editor(data, *args, key=None, **kwargs):
        if key in overrides:
            data = pd.DataFrame(overrides[key])
        return original(data, *args, key=key, **kwargs)

    def download_button(label, data, *args, **kwargs):
        st.session_state.setdefault("_downloads", {})[label] = data
        return original_download(label, data, *args, **kwargs)

    st.data_editor, st.download_button = data_editor, download_button
    try:
        runpy.run_path(f"{root}/app.py", run_name="__main__")
    finally:
        st.data_editor, st.download_button = original, original_download


def _app_with_tables(overrides):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_function(_patched_app_script, args=(str(ROOT), overrides), default_timeout=60)


def test_app_weak_axis_suggestion_keeps_orientation():
    """B1: viga simple 4 m, q = 3 kN/m (D) + peso propio, IPE 200 en eje débil → FLUENCIA (141 %).

    La búsqueda debe recorrer el catálogo EN EJE DÉBIL: σ_u = M_u·c/I_y con M_u = 1,4·(q + g)·L²/8 lleva a
    IPE 270 (antes sugería IPE 140, calculado en eje fuerte, que en eje débil da 316 %).
    """
    pytest.importorskip("streamlit")
    from core.builders import BeamDistLoad, BeamSupport, build_beam
    from core.design import suggest_section
    from core.materials import DEFAULT_MATERIAL, MATERIALS, get_section
    from core.model import SupportType
    from core.verification import check_combinations

    span, q = 4000.0, -3.0

    def build(sec):
        return build_beam(span, [BeamSupport(0.0, SupportType.PINNED), BeamSupport(span, SupportType.ROLLER)],
                          sec, MATERIALS[DEFAULT_MATERIAL], [], [BeamDistLoad(0.0, span, q, q)],
                          self_weight=True)

    ref = suggest_section(lambda s: build(s.weak_axis()), "IPE", 1.5, phi=0.9)
    assert ref is not None and ref.section.name == "IPE 270"
    assert not check_combinations(build(get_section("IPE 140").weak_axis())).strength_ok

    at = _app_with_tables({
        "point_loads": {"x [m]": [], "P [kN] ↓": [], "M [kN·m] ↺": [], "Hipótesis": []},
        "dist_loads": {"x ini [m]": [0.0], "x fin [m]": [4.0], "q ini [kN/m] ↓": [3.0],
                       "q fin [kN/m] ↓": [3.0], "Hipótesis": ["D"]},
    }).run()
    at.sidebar.number_input[0].set_value(4.0)
    at.sidebar.selectbox(key="beam_orientation").set_value("Eje débil")
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    badge = next(md.value for md in at.markdown if "border-left:6px" in md.value)
    assert ">FLUENCIA<" in badge and "ELU 141%" in badge
    labels = [m.label for m in at.metric]
    assert "IPE 270 (eje débil)" in labels and "IPE 140" not in labels
    expected = "IPE 200 (eje débil) no alcanza: subir a **IPE 270 (eje débil)**."
    assert any(w.value == expected for w in at.warning)


def test_app_weak_axis_current_profile_compared_by_base_name():
    """B1: si el perfil elegido en eje débil ya es el más liviano que verifica, se reconoce aunque su nombre
    lleve el sufijo "(eje débil)" (comparación por nombre base de catálogo)."""
    pytest.importorskip("streamlit")
    at = _app_with_tables({
        "point_loads": {"x [m]": [], "P [kN] ↓": [], "M [kN·m] ↺": [], "Hipótesis": []},
        "dist_loads": {"x ini [m]": [0.0], "x fin [m]": [4.0], "q ini [kN/m] ↓": [3.0],
                       "q fin [kN/m] ↓": [3.0], "Hipótesis": ["D"]},
    }).run()
    at.sidebar.number_input[0].set_value(4.0)
    at.sidebar.selectbox(key="beam_name").set_value("IPE 270")
    at.sidebar.selectbox(key="beam_orientation").set_value("Eje débil")
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    assert any("ya es el más liviano" in s.value for s in at.success)


def test_app_pdf_cache_key_includes_stability_k_and_lb():
    """B3: activar la estabilidad o cambiar K / Lb cambia la memoria: el PDF no puede salir de la caché del
    modelo anterior (antes la clave era modelo | L/N | escala y devolvía el PDF sin §4 de estabilidad)."""
    pytest.importorskip("streamlit")
    pypdf = pytest.importorskip("pypdf")
    import hashlib
    import io

    def report(at):
        assert not at.exception, [e.message for e in at.exception]
        pdf = at.session_state["_downloads"]["Descargar memoria"]
        text = " ".join(" ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages).split())
        return hashlib.sha256(pdf).hexdigest(), text

    at = _app_with_tables({}).run()
    h_off, pdf_off = report(at)
    markers = ("Estabilidad: pandeo flexional según", "Resistencia de diseño a compresión")  # §1 y §4
    assert not any(mk in pdf_off for mk in markers)

    next(c for c in at.sidebar.checkbox if c.label.startswith("Verificar estabilidad")).check().run()
    h_on, pdf_on = report(at)
    assert h_on != h_off
    assert all(mk in pdf_on for mk in markers) and "K = 1," in pdf_on

    k_input = next(n for n in at.sidebar.number_input if n.label == "Factor de longitud efectiva K")
    k_input.set_value(2.0).run()
    h_k, pdf_k = report(at)
    assert h_k != h_on and "K = 2," in pdf_k

    next(n for n in at.sidebar.number_input if n.label.startswith("Lb no arriostrada")).set_value(1.5).run()
    h_lb, pdf_lb = report(at)
    assert h_lb != h_k and "Lb = 1.500 m" in pdf_lb


_GENERIC_FRAME = {
    # Pórtico a dos aguas de 5 nodos + una fila recién agregada y vacía (como la deja el editor dinámico)
    "generic_nodes": {"x [m]": [0.0, 0.0, 4.0, 8.0, 8.0, None], "y [m]": [0.0, 4.0, 5.5, 4.0, 0.0, None]},
    "generic_members": {"i": [0, 1, 2, 3], "j": [1, 2, 3, 4], "Rótula inicio": [False] * 4,
                        "Rótula fin": [False] * 4},
    "generic_supports": {"Nodo": [0, 4], "Tipo": ["empotrado", "empotrado"]},  # SupportType.FIXED.value
    "generic_nodal_loads": {"Nodo": [2], "Fx [kN]": [5.0], "Fy [kN] ↓": [20.0], "Mz [kN·m]": [0.0],
                            "Hipótesis": ["D"]},
}


def test_app_generic_frame_more_than_four_nodes():
    """B5: el editor de nodos del pórtico genérico acepta más de 4 nodos; el número de nodo es la posición
    en la tabla (sin columna "Nodo" que `dropna` descarte) y la fila vacía final se ignora."""
    pytest.importorskip("streamlit")
    at = _app_with_tables(_GENERIC_FRAME).run()
    at.sidebar.radio[0].set_value("Pórtico genérico").run()

    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any(c.value == "Nodos: 0 (0; 0) · 1 (0; 4) · 2 (4; 5.5) · 3 (8; 4) · 4 (8; 0)"
               for c in at.sidebar.caption)
    reactions = next(d.value for d in at.dataframe if "Rx [kN]" in d.value.columns)
    assert list(reactions["Nodo"]) == [0, 4]
    assert list(reactions["x [m]"]) == [0.0, 8.0]
    # Equilibrio de la combinación gobernante (sólo D): ΣRx = −1,4·Fx
    assert abs(reactions["Rx [kN]"].sum() + 1.4 * 5.0) < 1e-6


def test_app_generic_frame_incomplete_middle_node_is_reported():
    """B5: una fila intermedia sin coordenadas NO se descarta (renumeraría los nodos siguientes y las barras
    quedarían conectadas a otros nodos): se informa con el número de nodo."""
    pytest.importorskip("streamlit")
    tables = dict(_GENERIC_FRAME)
    tables["generic_nodes"] = {"x [m]": [0.0, 0.0, None, 8.0, 8.0], "y [m]": [0.0, 4.0, None, 4.0, 0.0]}
    at = _app_with_tables(tables).run()
    at.sidebar.radio[0].set_value("Pórtico genérico").run()

    assert not at.exception, [e.message for e in at.exception]
    assert any("Nodo(s) 2 sin coordenadas" in e.value for e in at.error)


def test_node_table_numbering_by_position():
    """B5: `_node_table` numera por posición; ignora sólo filas vacías del final, conserva las intermedias."""
    pytest.importorskip("streamlit")
    import pandas as pd

    from ui.inputs import _node_table

    df = pd.DataFrame({"x [m]": [0.0, None, 8.0, None, None], "y [m]": [0.0, None, 4.0, None, None]},
                      index=[0, 1, 3, 7, 8])  # índices con huecos, como tras borrar filas
    out = _node_table(df)
    assert list(out.index) == [0, 1, 2]
    assert out["x [m]"].iloc[2] == 8.0 and pd.isna(out["x [m]"].iloc[1])
