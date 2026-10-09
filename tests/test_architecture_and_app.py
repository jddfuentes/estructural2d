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
