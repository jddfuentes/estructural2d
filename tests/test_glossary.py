"""Glosario: fuente única (core/glossary.py) para la memoria PDF y la wiki de la app."""

from __future__ import annotations

import datetime as dt
import io
import re

import pytest

from core.builders import build_portal_frame
from core.glossary import GRAPHIC_SYMBOLS, NOTATION, TERMS, to_markdown, to_tex
from core.materials import MATERIALS, get_section
from core.model import SupportType as ST
from core.reports import build_pdf_report
from core.solver import solve
from core.verification import check_safety


@pytest.mark.parametrize(
    "symbol,tex",
    [
        ("σ_{máx}", r"\sigma _{\text{máx}}"),
        ("FS_{adm}", r"\text{FS}_{\text{adm}}"),
        ("F_{x}, F_{y}", "F_{x}, F_{y}"),
        ("cm^{4}", "cm^{4}"),
        ("|σ|_{máx}", r"\vert \sigma \vert _{\text{máx}}"),
    ],
)
def test_to_tex(symbol, tex):
    assert to_tex(symbol) == tex


def test_markdown_is_complete_and_well_formed():
    md = to_markdown()
    for g in NOTATION:
        assert f"**{g.title}**" in md
    for gs in GRAPHIC_SYMBOLS:
        assert f"**{gs.name}**" in md
    for t in TERMS:
        assert f"**{t.term}**" in md
    for line in md.splitlines():
        if line.startswith("|"):
            assert line.count("|") in (3, 4), line  # 2 o 3 columnas: ningún '|' suelto rompe la tabla
        assert line.count("$") % 2 == 0, line  # LaTeX inline balanceado
        outside_math = re.sub(r"\$[^$]*\$", "", line)
        assert "_{" not in outside_math and "^{" not in outside_math, line  # mini-notación convertida


def test_pdf_glossary_comes_from_core_glossary():
    """Todo lo definido en core/glossary.py aparece en la sección 6 del PDF."""
    pypdf = pytest.importorskip("pypdf")
    mat = MATERIALS["ASTM A36"]
    m = build_portal_frame(8000.0, 4000.0, get_section("IPE 240"), get_section("IPE 300"), mat,
                           ST.FIXED, ST.PINNED, beam_q=-12.0)
    r = solve(m)
    pdf = build_pdf_report(m, r, check_safety(m, r), date=dt.date(2026, 10, 9))
    text = " ".join(" ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages).split())
    glossary = text[text.index("6. Glosario y simbología"):]
    for g in NOTATION:
        assert g.title in glossary
    for gs in GRAPHIC_SYMBOLS:
        assert gs.name in glossary
    for t in TERMS:
        assert t.term in glossary
    assert "Ejes locales de barra: x' del nodo i al nodo j" in glossary
