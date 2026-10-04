# Estructural 2D — predimensionamiento rápido

App Streamlit para cálculo elástico lineal de vigas y pórticos planos: diagramas N-V-M,
deformada, tensión normal máxima, factor de seguridad y búsqueda del perfil más liviano.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                 # o requirements.txt sólo para correr
streamlit run app.py
pytest && mypy && ruff check .
```

- Motor: método directo de rigidez propio (NumPy), exacto por barra para Euler-Bernoulli
  con cargas lineales; validado contra fórmulas clásicas y contra `anastruct`.
- Unidades del core: mm, N, MPa. La UI trabaja en m, kN, kN/m, kN·m.
- Roles, convenciones y protocolo de cambios: ver [AGENTS.md](AGENTS.md).

> Herramienta de predimensionamiento: no considera pandeo, corte, fatiga ni efectos de
> segundo orden. Verificar perfiles contra el catálogo del proveedor.
