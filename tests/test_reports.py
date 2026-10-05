"""Memoria de cálculo PDF (core/reports.py)."""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam, build_portal_frame
from core.materials import MATERIALS, get_section
from core.model import SupportType as ST
from core.reports import build_pdf_report
from core.solver import solve
from core.verification import Status, check_safety

MAT = MATERIALS["ASTM A36"]
DATE = dt.date(2026, 10, 4)


def _beam(q: float = -10.0, P: float = -20_000.0, section: str = "IPE 200"):
    m = build_beam(6000.0, [BeamSupport(0, ST.PINNED), BeamSupport(6000.0, ST.ROLLER)],
                   get_section(section), MAT,
                   point_loads=[BeamPointLoad(3000.0, Fy=P)],
                   dist_loads=[BeamDistLoad(0.0, 6000.0, q, q)], self_weight=True)
    r = solve(m)
    return m, r, check_safety(m, r)


def _portal():
    m = build_portal_frame(8000.0, 4000.0, get_section("IPE 240"), get_section("IPE 300"), MAT,
                           ST.FIXED, ST.PINNED, beam_q=-12.0, lateral_load=10_000.0, self_weight=True)
    r = solve(m)
    return m, r, check_safety(m, r)


def _text(pdf: bytes) -> str:
    pypdf = pytest.importorskip("pypdf")
    import io

    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return "\n".join(p.extract_text() for p in reader.pages)


def test_returns_valid_pdf_bytes():
    m, r, c = _beam()
    pdf = build_pdf_report(m, r, c, "Viga de prueba", "J. Fuentes", date=DATE)
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 2_000


@pytest.mark.parametrize("builder", [_beam, _portal])
def test_report_content(builder):
    m, r, c = builder()
    text = _text(build_pdf_report(m, r, c, "Proyecto X", "Autor Y", date=DATE))
    for key in ("Proyecto X", "Autor Y", "04/10/2026", "Reacciones", "Solicitaciones máximas",
                "Factor de seguridad", "VEREDICTO", c.status.value, "ASTM A36"):
        assert key in text, key
    for mem in m.members:
        assert mem.section.name in text


@pytest.mark.parametrize("q,status", [(-3.0, Status.OK), (-10.0, Status.ALERT), (-25.0, Status.FAIL)])
def test_verdict_matches_check(q, status):
    m, r, c = _beam(q, P=0.0)
    assert c.status is status
    assert f"VEREDICTO: {status.value}" in _text(build_pdf_report(m, r, c, date=DATE))


def test_key_numbers_in_report():
    """FS y σ del PDF coinciden con SafetyCheck; flecha relativa con la longitud de referencia."""
    m, r, c = _beam(-3.0, P=0.0)
    text = _text(build_pdf_report(m, r, c, date=DATE, reference_length=6000.0))
    assert f"{c.fs:.2f}" in text
    assert f"{c.sigma_max:,.1f} MPa" in text
    d = abs(r.extreme("deflection").value)
    assert f"L/{6000.0 / d:,.0f}" in text


def test_is_pure_and_deterministic():
    m, r, c = _portal()
    m_before = copy.deepcopy(m)
    a = build_pdf_report(m, r, c, "T", "A", date=DATE)
    b = build_pdf_report(m, r, c, "T", "A", date=DATE)
    assert a == b  # PDF invariant: sin timestamps ni IDs aleatorios
    assert m.nodes == m_before.nodes and m.distributed_loads == m_before.distributed_loads


def test_escapes_user_text():
    """Texto libre con caracteres XML no debe romper el parser de párrafos de ReportLab."""
    m, r, c = _beam()
    pdf = build_pdf_report(m, r, c, "Viga <A&B> \"test\"", "Juan & Cía <SRL>", date=DATE)
    assert pdf.startswith(b"%PDF")
    assert "Juan & Cía <SRL>" in _text(pdf)


def test_empty_title_and_default_date():
    m, r, c = _beam()
    pdf = build_pdf_report(m, r, c, "   ", "")
    assert pdf.startswith(b"%PDF")
    assert f"{dt.date.today():%d/%m/%Y}" in _text(pdf)
