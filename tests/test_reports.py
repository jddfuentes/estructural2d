"""Memoria de cálculo PDF (core/reports.py)."""

from __future__ import annotations

import base64
import copy
import datetime as dt
from dataclasses import replace

import pytest

from core.builders import BeamDistLoad, BeamPointLoad, BeamSupport, build_beam, build_portal_frame
from core.materials import MATERIALS, get_section
from core.model import SupportType as ST
from core.reports import build_pdf_report
from core.solver import solve
from core.verification import Status, check_safety

MAT = MATERIALS["ASTM A36"]
DATE = dt.date(2026, 10, 4)
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


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


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_channel_section_notes_and_warnings_in_report():
    """Perfil UPN: nota de fuente en 2.4 y advertencia de torsión en 5; ausentes con IPE."""
    m, r, c = _beam(section="UPN 200")
    text = _flat(_text(build_pdf_report(m, r, c, date=DATE)))
    body = text[:text.index("6. Glosario y simbología")]
    assert "DIN 1026-1" in body
    assert "Sección en canal (U/C)" in body

    m, r, c = _beam()
    body = _flat(_text(build_pdf_report(m, r, c, date=DATE)))
    body = body[:body.index("6. Glosario y simbología")]
    assert "Euronorm 19-57" in body
    assert "Sección en canal (U/C)" not in body


def test_returns_valid_pdf_bytes():
    m, r, c = _beam()
    pdf = build_pdf_report(m, r, c, "Viga de prueba", "J. Fuentes", date=DATE)
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 2_000


def test_report_with_internal_releases_returns_valid_pdf():
    """Las rótulas de extremo se incluyen en geometría y simbología sin invalidar el PDF."""
    m, r, c = _beam(P=0.0)
    m.members[0] = replace(m.members[0], release_end=True)
    m.members[1] = replace(m.members[1], release_start=True)
    assert build_pdf_report(m, r, c, date=DATE).startswith(b"%PDF")


@pytest.mark.parametrize("image_bytes", [None, PNG_1X1])
def test_report_accepts_optional_embedded_plot(image_bytes: bytes | None):
    m, r, c = _beam()
    pdf = build_pdf_report(m, r, c, date=DATE, image_bytes=image_bytes)
    assert pdf.startswith(b"%PDF")


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


def test_deflection_alert_in_report():
    """IPE 240, L = 8 m, q = 5 N/mm: FS = 2,03 pero δ = 5qL⁴/384EI = 34,3 mm > L/300 ⇒ ALERTA por flecha."""
    L = 8000.0
    m = build_beam(L, [BeamSupport(0, ST.PINNED), BeamSupport(L, ST.ROLLER)], get_section("IPE 240"), MAT,
                   dist_loads=[BeamDistLoad(0.0, L, -5.0, -5.0)])
    r = solve(m)
    c = check_safety(m, r)
    assert c.status is Status.ALERT and c.strength_ok and not c.deflection_ok
    text = " ".join(_text(build_pdf_report(m, r, c, date=DATE)).split())
    assert "VEREDICTO: ALERTA — flecha mayor a la admisible (servicio)" in text
    assert "FS menor al admisible" not in text
    assert f"L/300 = {L / 300:,.2f} mm" in text
    assert f"{c.delta_max:,.2f} mm" in text


def test_key_numbers_in_report():
    """FS y σ del PDF coinciden con SafetyCheck; flecha relativa con la longitud de referencia."""
    m, r, c = _beam(-3.0, P=0.0)
    text = _text(build_pdf_report(m, r, c, date=DATE, reference_length=6000.0))
    assert f"{c.fs:.2f}" in text
    assert f"{c.sigma_max:,.1f} MPa" in text
    d = abs(r.extreme("deflection").value)
    assert f"L/{6000.0 / d:,.0f}" in text


def test_rejects_non_positive_reference_length():
    m, r, c = _beam()
    with pytest.raises(ValueError, match="longitud de referencia"):
        build_pdf_report(m, r, c, date=DATE, reference_length=-500.0)


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


def test_glossary_section():
    """Sección 6: notación con unidades, símbolos gráficos y términos; desactivable."""
    m, r, c = _portal()
    text = " ".join(_text(build_pdf_report(m, r, c, date=DATE)).split())
    for key in ("6. Glosario y simbología", "6.1 Notación", "6.2 Símbolos gráficos", "6.3 Términos",
                "Ejes locales de barra: x' del nodo i al nodo j", "Módulo resistente elástico, W = I /",
                "Empotrado", "Articulado", "Móvil", "Lado traccionado", "sección 6 (Glosario y simbología)"):
        assert key in text, key
    without = _text(build_pdf_report(m, r, c, date=DATE, include_glossary=False))
    assert "Glosario" not in without


# --------------------------------------------------------------------------- #
# Memoria con CombinationCheck (LRFD por combinaciones D / L)
# --------------------------------------------------------------------------- #

_L = 6000.0


def _combo_beam(section, q_D: float, q_L: float, self_weight: bool = True):
    """Viga simple: q_D (D) + q_L (L) uniformes; devuelve (modelo, CombinationCheck)."""
    from core.model import LoadCase
    from core.verification import check_combinations

    m = build_beam(_L, [BeamSupport(0, ST.PINNED), BeamSupport(_L, ST.ROLLER)], section, MAT,
                   dist_loads=[BeamDistLoad(0.0, _L, q_D, q_D),
                               BeamDistLoad(0.0, _L, q_L, q_L, case=LoadCase.LIVE)],
                   self_weight=self_weight)
    return m, check_combinations(m)


def _slice(text: str, start: str, end: str) -> str:
    return text[text.index(start):text.index(end, text.index(start))]


def test_combination_report_service_alert_is_consistent():
    """IPE 220, q_D = 2 + peso propio, q_L = 6 N/mm, L = 6 m: ELU verifica y ELS no (δ > L/300).

    Portada y §1 documentan LRFD (σu ≤ φ·Sy) y el límite L/300; §3.3 informa la flecha de servicio
    1.0D + 1.0L (no la mayorada); §4 da ALERTA con los motivos de `check.issues`; el equilibrio de
    §3.1 se calcula con las cargas mayoradas de la combinación gobernante (residuo ≈ 0).
    """
    import re

    m, chk = _combo_beam(get_section("IPE 220"), -2.0, -6.0)
    assert chk.strength_ok and not chk.deflection_ok and chk.verdict == (Status.ALERT, "ALERTA")
    text = _flat(_text(build_pdf_report(m, chk.governing_results, chk, date=DATE)))

    cover = text[:text.index("1. Bases de cálculo")]
    assert "LRFD" in cover and "L/300" in cover and "FS ≥" not in cover
    basis = _slice(text, "1. Bases de cálculo", "2. Datos de entrada")
    assert "sin límite" not in basis and "FS" not in basis
    assert "φ = 0.9" in basis and "L/300" in basis and "1.0D + 1.0L" in basis

    deformation = _slice(text, "3.3 Deformación de servicio (1.0D + 1.0L)", "4. Verificación")
    sv = chk.service_check
    assert f"{sv.delta_max:,.2f} mm" in deformation
    assert f"L/{sv.deflection_ratio:,.0f}" in deformation
    delta_u = chk.governing_results.extreme("deflection").value
    assert f"{abs(delta_u):,.2f} mm" not in deformation  # nunca la flecha mayorada

    found = re.search(r"Residuo relativo \|ΣR \+ ΣF\| / \|F\| = ([0-9.]+e[+-][0-9]+)", text)
    assert found is not None
    residual = float(found.group(1))
    assert residual < 1e-9
    assert "cargas mayoradas (1.2D + 1.6L)" in text or "cargas mayoradas (1.4D)" in text

    verdict = text[text.index("VEREDICTO"):text.index("5. Limitaciones")]
    assert verdict.startswith("VEREDICTO: ALERTA") and "OK — VERIFICA" not in text
    assert "ELS 1.0D + 1.0L" in verdict


@pytest.mark.parametrize(("factor", "label"), [(1.0, "NO VERIFICA"), (2.0, "FLUENCIA")])
def test_combination_report_uls_failure(factor, label):
    """σu(1.2D + 1.6L) ≈ 0.95·Sy (φ·Sy < σu ≤ Sy) ⇒ NO VERIFICA; con el doble de L, σu > Sy ⇒ FLUENCIA.

    σu = (1.2·|q_D| + 1.6·|q_L|)·L²/8 · c/I (sección propia, sin peso propio). Ambos en rojo (FAIL).
    """
    from core.materials import custom

    I, c, sy = 2.0e7, 100.0, MAT.Sy
    q_L = -((0.95 * sy * I / c * 8 / _L**2) - 1.2 * 5.0) / 1.6
    m, chk = _combo_beam(custom("t", A=3000.0, I=I, c=c), -5.0, factor * q_L, self_weight=False)
    assert chk.verdict == (Status.FAIL, label)
    text = _flat(_text(build_pdf_report(m, chk.governing_results, chk, date=DATE)))
    verdict = text[text.index("VEREDICTO"):text.index("5. Limitaciones")]
    assert verdict.startswith(f"VEREDICTO: {label}")
    assert "ELU 1.2D + 1.6L" in verdict


def test_combination_report_ok():
    m, chk = _combo_beam(get_section("IPE 300"), -2.0, -3.0)
    assert chk.verdict == (Status.OK, "VERIFICA")
    text = _flat(_text(build_pdf_report(m, chk.governing_results, chk, date=DATE)))
    assert "VEREDICTO: OK — VERIFICA" in text


# --------------------------------------------------------------------------- #
# Resortes y cargas por hipótesis en la memoria
# --------------------------------------------------------------------------- #


def test_report_springs_units_and_inert_components():
    """Tabla 2.2: Kx en kN/m (N/mm ≡ kN/m, sin factor) y Krz en kN·m/rad (×1e-6); un Kx sobre un
    articulado (ux ya restringido) se informa "sin efecto" y no figura como rigidez activa."""
    from core.model import Spring

    m, _, _ = _beam(P=0.0)
    m.supports = [replace(m.supports[0]), replace(m.supports[1], type=ST.FREE)]
    last = len(m.nodes) - 1
    m.springs = [Spring(0, kx=2500.0, krz=4.0e9), Spring(last, ky=5000.0)]
    r = solve(m)
    text = _flat(_text(build_pdf_report(m, r, check_safety(m, r), date=DATE)))
    supports = _slice(text, "2.2 Condiciones de apoyo", "2.3 Cargas aplicadas")
    assert "Krz=4000 kN·m/rad" in supports and "Kx sin efecto" in supports
    assert "Kx=2500" not in supports and "Kx=2.5 kN/m" not in supports
    assert "Elástico: Ky=5000 kN/m" in supports  # libre + resorte vertical: 5000 N/mm = 5000 kN/m


def test_report_load_labels_by_case():
    """Con hipótesis L, el esquema rotula qD / qL; sin L, el rótulo sigue siendo q."""
    from core.model import LoadCase

    m = build_beam(6000.0, [BeamSupport(0, ST.PINNED), BeamSupport(6000.0, ST.ROLLER)],
                   get_section("IPE 300"), MAT,
                   point_loads=[BeamPointLoad(3000.0, Fy=-20_000.0, case=LoadCase.LIVE)],
                   dist_loads=[BeamDistLoad(0.0, 6000.0, -8.0, -8.0),
                               BeamDistLoad(0.0, 6000.0, -5.0, -5.0, case=LoadCase.LIVE)])
    r = solve(m)
    text = _flat(_text(build_pdf_report(m, r, check_safety(m, r), date=DATE)))
    assert "qD = 8 kN/m" in text and "qL = 5 kN/m" in text and "20.0 kN (L)" in text
    m0, r0, c0 = _beam()
    text0 = _flat(_text(build_pdf_report(m0, r0, c0, date=DATE)))
    assert "q = 10 kN/m" in text0 and "qD =" not in text0


def _beam_with_springs():
    """Viga 0—1—2: articulado + Krz en 0, sólo resorte Ky en 1 (sin `Support`), móvil en 2."""
    from core.model import DistributedLoad, Member, Model, Spring, Support

    sec = get_section("IPE 200")
    m = Model(
        nodes=[(0.0, 0.0), (3000.0, 0.0), (6000.0, 0.0)],
        members=[Member(0, 1, sec, MAT), Member(1, 2, sec, MAT)],
        supports=[Support(0, ST.PINNED), Support(2, ST.ROLLER)],
        distributed_loads=[DistributedLoad(0, -10.0, -10.0), DistributedLoad(1, -10.0, -10.0)],
        springs=[Spring(1, ky=1000.0), Spring(0, krz=5.0e9)],
    )
    r = solve(m)
    return m, r, check_safety(m, r)


def test_report_spring_stiffness_units_and_labels():
    """2.2 Apoyos: Ky = 1000 N/mm se imprime 1000 kN/m (N/mm = kN/m) y Krz = 5e9 N·mm/rad = 5000 kN·m/rad.

    Un nodo sólo con resorte se rotula "Elástico: …" (sin duplicar) y figura en 3.1 Reacciones.
    """
    m, r, c = _beam_with_springs()
    text = _flat(_text(build_pdf_report(m, r, c, date=DATE)))
    assert "0 0.000 0.000 Articulado + Elástico: Krz=5000 kN·m/rad ux, uy" in text  # 2.2
    assert "1 3.000 0.000 Elástico: Ky=1000 kN/m —" in text
    assert "ElásticoElástico" not in text
    reactions = text[text.index("3.1 Reacciones"):text.index("3.2 ")]
    assert "0 Articulado + elástico" in reactions and "1 Elástico" in reactions


def test_report_free_support_with_springs():
    """Pórtico con bases "libre" + resortes: se rotulan como elásticas, sin símbolo rígido."""
    from core.model import Spring

    m = build_portal_frame(8000.0, 4000.0, get_section("IPE 240"), get_section("IPE 300"), MAT,
                           ST.FREE, ST.FREE, beam_q=-12.0, lateral_load=10_000.0)
    m.springs = [Spring(0, kx=2.0e4, ky=5.0e5, krz=3.0e10), Spring(3, kx=2.0e4, ky=5.0e5, krz=3.0e10)]
    r = solve(m)
    text = _flat(_text(build_pdf_report(m, r, check_safety(m, r), date=DATE)))
    assert "Elástico: Kx=20000 kN/m, Ky=500000 kN/m, Krz=30000 kN·m/rad" in text
    assert "Libre" not in text[:text.index("6. Glosario y simbología")]
