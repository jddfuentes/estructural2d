"""Wiki técnica de ingeniería del modelo estructural."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Wiki de ingeniería · Estructural 2D", page_icon="📚", layout="wide")

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "engineering_wiki.md"

st.title("Wiki de ingeniería")
st.caption("Ecuaciones, hipótesis, criterios y registro de validación técnica")

if not DOC_PATH.is_file():
    st.error("No se encontró el documento técnico de la wiki.")
    st.stop()

st.markdown(DOC_PATH.read_text(encoding="utf-8"))
