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
