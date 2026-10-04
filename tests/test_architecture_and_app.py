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
