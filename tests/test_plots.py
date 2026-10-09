"""Pruebas de presentación de las figuras Plotly."""

from __future__ import annotations

from core.builders import BeamSupport, build_beam
from core.materials import Material, custom
from core.model import SupportType
from ui.plots import plot_structure


def test_structure_plot_shows_global_reference_axes() -> None:
    material = Material("test", E=200_000.0, Sy=250.0, Su=400.0)
    section = custom("test", A=3_000.0, I=2.0e7, c=100.0)
    model = build_beam(
        6_000.0,
        [BeamSupport(0.0, SupportType.PINNED), BeamSupport(6_000.0, SupportType.ROLLER)],
        section,
        material,
    )

    figure = plot_structure(model, show_loads=False)
    texts = [annotation.text or "" for annotation in figure.layout.annotations]

    assert any("+X" in text for text in texts)
    assert any("+Y" in text for text in texts)
    assert any("+Z" in text and "+Mz" in text for text in texts)
