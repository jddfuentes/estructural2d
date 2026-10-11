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
