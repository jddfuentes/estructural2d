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


# --------------------------------------------------------------------------- #
# Encabezado, tipografía y figuras
# --------------------------------------------------------------------------- #


def _pages(pdf: bytes) -> list[str]:
    pypdf = pytest.importorskip("pypdf")
    import io

    return [p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages]


def test_default_title_not_duplicated():
    """Con el título por defecto, 'memoria de cálculo' aparece una sola vez en la portada."""
    m, r, c = _beam()
    first = _pages(build_pdf_report(m, r, c, date=DATE))[0]
    assert first.casefold().count("memoria de cálculo") == 1


def test_custom_title_once_per_page():
    """Título grande en la pág. 1; en las siguientes, sólo en el encabezado."""
    m, r, c = _portal()
    pages = _pages(build_pdf_report(m, r, c, "Pórtico nave de bombas", date=DATE))
    assert len(pages) >= 2
    assert all(p.count("Pórtico nave de bombas") == 1 for p in pages)


def test_unicode_font_embedded():
    """DejaVu Sans embebida: kN·m (U+00B7) y σ/δ se imprimen como texto real en cualquier visor."""
    m, r, c = _beam()
    pdf = build_pdf_report(m, r, c, date=DATE)
    assert b"DejaVuSans" in pdf
    text = _text(pdf)
    assert "kN·m" in text and "kN-m" not in text
    assert "σ" in text and "δ" in text


@pytest.mark.parametrize("builder,n_figs", [(_beam, 6), (_portal, 7)])
def test_figures_numbered(builder, n_figs):
    """Convenciones + esquema + diagramas + deformada.

    Viga: M, V, σ (N nulo) -> 6 figuras. Pórtico: M, V, N, σ -> 7 figuras.
    """
    m, r, c = builder()
    text = _text(build_pdf_report(m, r, c, date=DATE))
    assert f"Figura {n_figs}." in text
    assert f"Figura {n_figs + 1}." not in text
    for key in ("Esquema estático", "Deformada amplificada", "Diagrama de momento flector M"):
        assert key in text


def test_null_diagram_reported_not_drawn():
    m, r, c = _beam()
    text = _text(build_pdf_report(m, r, c, date=DATE))
    assert "Esfuerzo normal N: nulo en toda la estructura" in text
    assert "Diagrama de esfuerzo normal N" not in text


def test_figures_are_vector_and_optional():
    """Figuras vectoriales (sin imágenes rasterizadas) y desactivables."""
    m, r, c = _portal()
    with_figs = build_pdf_report(m, r, c, date=DATE)
    without = build_pdf_report(m, r, c, date=DATE, include_figures=False)
    assert b"/Subtype /Image" not in with_figs
    assert "Figura" not in _text(without)
    assert len(without) < len(with_figs)


def test_sign_references_present():
    """Referencias de signo: figura de convenciones, ejes globales y locales x'-y' en pórticos."""
    m, r, c = _portal()
    text = " ".join(_text(build_pdf_report(m, r, c, date=DATE)).split())  # epígrafes partidos en líneas
    for key in ("Convenciones de signos", "N > 0: tracción", "M > 0: tracción inferior", "+Mz",
                "Ejes locales de barra", "x'", "y'", "Signo según ejes locales x'"):
        assert key in text, key
    mb, rb, cb = _beam()
    beam_text = _text(build_pdf_report(mb, rb, cb, date=DATE))
    assert "+M" in beam_text and "+V" in beam_text  # ejes con sentido positivo en los diagramas
