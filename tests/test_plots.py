"""Pruebas de presentación de las figuras Plotly."""

from __future__ import annotations

from core.builders import BeamSupport, build_beam
from core.materials import Material, custom
from core.model import SupportType
from core.solver import solve
from ui.plots import plot_diagram, plot_structure


def test_structure_plot_shows_global_reference_axes() -> None:
    material = Material("test", E=200_000.0, Sy=250.0, Su=400.0)
    section = custom("test", A=3_000.0, I=2.0e7, c=100.0)
    model = build_beam(
        6_000.0,
        [BeamSupport(0.0, SupportType.PINNED), BeamSupport(6_000.0, SupportType.ROLLER)],
        section,
        material,
    )

    figure = plot_structure(model, solve(model), show_loads=False)
    texts = [annotation.text or "" for annotation in figure.layout.annotations]

    assert any("+X" in text for text in texts)
    assert any("+Y" in text for text in texts)
    assert any("+Z" in text and "+Mz" in text for text in texts)


def test_structure_plot_spring_labels_units_and_side() -> None:
    """Hover del resorte en kN/m y kN·m/rad; el resorte Kx de la base derecha se dibuja hacia afuera."""
    from core.builders import build_portal_frame
    from core.model import Spring

    material = Material("test", E=200_000.0, Sy=250.0, Su=400.0)
    section = custom("test", A=3_000.0, I=2.0e7, c=100.0)
    model = build_portal_frame(6_000.0, 4_000.0, section, section, material,
                               SupportType.FREE, SupportType.FREE, beam_q=-10.0)
    model.springs = [Spring(0, kx=2.0e4, ky=5.0e5, krz=3.0e10), Spring(3, kx=1.0e3)]

    figure = plot_structure(model, solve(model), show_loads=False)
    hovers = " ".join(str(t.customdata) for t in figure.data if t.customdata is not None)
    assert "Kx = 20000 kN/m" in hovers and "Ky = 500000 kN/m" in hovers
    assert "Krz = 30000 kN·m/rad" in hovers and "Kx = 1000 kN/m" in hovers

    x_right = model.nodes[3][0] * 1e-3  # m
    spring_traces = [t for t in figure.data if t.line is not None and t.line.color == "#B45309"]
    right = [t for t in spring_traces if min(t.x) >= x_right - 1e-9 and max(t.x) > x_right]
    assert right, "el resorte Kx del nodo derecho debe quedar a la derecha (hacia afuera)"


def test_stress_diagrams_render_shear_and_von_mises() -> None:
    material = Material("test", E=200_000.0, Sy=250.0, Su=400.0)
    section = custom("test", A=3_000.0, I=2.0e7, c=100.0, Q_max=25_000.0, t_web=8.0)
    model = build_beam(
        6_000.0,
        [BeamSupport(0.0, SupportType.PINNED), BeamSupport(6_000.0, SupportType.ROLLER)],
        section,
        material,
        dist_loads=[],
    )
    results = solve(model)
    for quantity, title in (("tau", "Tensión de corte"), ("sigma_vm", "von Mises")):
        figure = plot_diagram(model, results, quantity)
        assert title in figure.layout.title.text
        assert len(figure.data) > 0
