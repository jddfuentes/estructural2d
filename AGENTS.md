# AGENTS.md — Estructural 2D (MVP)

Guía operativa para agentes de IA (y humanos) que trabajan en este repositorio.
Leerla completa antes de tocar código. Si una instrucción de una tarea contradice
este archivo, gana este archivo salvo que la tarea diga explícitamente que lo modifica.

## 1. Qué es esto

App web (Streamlit) de **predimensionamiento rápido y cálculo elástico lineal 2D** de
vigas y pórticos planos para ingeniería mecánica/industrial. Método directo de rigidez,
elemento de pórtico Euler-Bernoulli, verificación por tensión normal con FS = Sy/σmax.

Filosofía: **tiempo de fin de semana, cero sobreingeniería**. Cada cambio tiene que
poder revisarse en 10 minutos y estar cubierto por un test analítico.

## 2. Comandos

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # runtime + test/tipado/lint
streamlit run app.py                                   # app en http://localhost:8501
pytest                                                 # tests (deben pasar 100 %)
mypy                                                   # tipado (strict en core/)
ruff check .                                           # lint (ruff check . --fix)
```

**Definición de "listo" para cualquier cambio:** `pytest && mypy && ruff check .` en verde.

## 3. Arquitectura y regla de dependencias

```
app.py            Orquestación Streamlit. NO calcula nada.
 ├── ui/inputs.py   Sidebar -> Model. ÚNICO lugar con conversión de unidades de UI.
 ├── ui/plots.py    Model/Results -> go.Figure (sin Streamlit; reutilizable en reportes)
 └── core/          Motor puro. Sin Streamlit, sin Plotly, sin pandas, sin I/O.
      ├── materials.py     Material, Section, catálogo (IPE, IPN, caños Sch40, tubos)
      ├── model.py         Model, Member, Support, NodalLoad, DistributedLoad (dataclasses)
      ├── builders.py      build_beam(), build_portal_frame(): definición "de ingeniero" -> Model
      ├── solver.py        solve(model) -> Results  (función pura)
      ├── verification.py  check_safety() -> SafetyCheck (FS, estado OK/ALERTA/FLUENCIA)
      └── design.py        suggest_section(): perfil más liviano que verifica
tests/            pytest, casos analíticos con fórmula de referencia
```

Regla dura (se verifica en revisión):

| Módulo | Puede importar | NO puede importar |
|---|---|---|
| `core/*` | stdlib, `numpy`, otros `core/*` | `streamlit`, `plotly`, `pandas`, `ui/*` |
| `ui/plots.py` | `core`, `numpy`, `plotly` | `streamlit` |
| `ui/inputs.py`, `app.py` | todo | — |

Chequeo rápido: `grep -rnE "streamlit|plotly|pandas" core/` debe devolver vacío.

## 4. Unidades y convenciones de signo

**Core: SI-mm sin excepciones.**

| Magnitud | Core | UI (sólo `ui/`) |
|---|---|---|
| Longitud | mm | m |
| Fuerza | N | kN |
| Carga distribuida | N/mm | kN/m (numéricamente igual) |
| Momento | N·mm | kN·m (×1e6) |
| Tensión, E | MPa (N/mm²) | MPa |
| Área / Inercia / W | mm² / mm⁴ / mm³ | cm² / cm⁴ / cm³ |
| Densidad | kg/m³ | kg/m³ |

- Ejes globales: X → derecha, Y ↑ arriba, Mz antihorario +.
- Cargas en el core: `Fy < 0` y `q < 0` son **hacia abajo**. La UI muestra "+ hacia abajo" y
  convierte el signo en `ui/inputs.py`.
- Ejes locales de barra: x de nodo `i` a `j`, y local = x rotado +90°.
- Solicitaciones: **N > 0 tracción**; **M > 0 tracciona la fibra de y local negativa**
  (en vigas horizontales: fibra inferior, "sagging"); **V = dM/dx**.
- Tensiones: `sigma_top = N/A − M·c_top/I`, `sigma_bot = N/A + M·c_bot/I`.
- Reacciones: en ejes globales, sobre la estructura.

Nombres físicos cortos (`E`, `I`, `A`, `L`, `M`, `V`, `N`, `q`) están permitidos y son
preferidos en fórmulas (E741 deshabilitado en ruff).

## 5. Contrato de API estable (no romper)

Estas firmas son la interfaz entre roles. Se pueden **agregar** parámetros con valor por
defecto y campos nuevos al final de dataclasses; **no** se pueden renombrar, quitar ni
cambiar semántica/unidades sin actualizar este archivo, los tests y todos los llamadores
en el mismo commit.

```python
core.solver.solve(model: Model, n_stations: int = 41) -> Results
core.solver.Results.extreme(quantity: str) -> Extreme      # "N","V","M","sigma","deflection","uy"
core.verification.check_safety(model, results, fs_min=1.5) -> SafetyCheck
core.builders.build_beam(...) / build_portal_frame(...) -> Model
core.design.suggest_section(build, family, fs_min, max_deflection=None) -> Suggestion | None
ui.plots.plot_structure(model, results=None, deformed_scale=None, ...) -> go.Figure
ui.plots.plot_diagram(model, results, quantity, moment_on_tension_side=True) -> go.Figure
```

Errores: datos inválidos → `ValueError`; estructura inestable → `core.solver.StructuralError`.
Mensajes en castellano y accionables (la UI los muestra tal cual).

## 6. Convenciones de código

- Python **3.10+**: `from __future__ import annotations`; nada de `StrEnum`, `typing.Self`,
  `type X = ...` ni `except*` (son ≥ 3.11). Enums de texto: `class X(str, Enum)`.
- Tipado completo; `mypy --strict` en `core/`. Arrays: `numpy.typing.NDArray[np.float64]`
  (alias `FloatArray` en `solver.py`).
- Datos: `@dataclass(frozen=True, slots=True)`; `Model` es mutable sólo durante su
  construcción en `builders`. **`solve()` nunca muta el modelo** (hay test).
- Sin estado global mutable, sin cachés ocultas en `core/`. Si hace falta cachear, se hace
  en la UI (`st.cache_data`).
- Docstrings y mensajes al usuario en **castellano**; identificadores en inglés.
- Cada constante normativa/tabla lleva comentario con su fuente (norma, catálogo).
- Línea máx. 110 caracteres. Sin dependencias nuevas sin justificar en el PR/commit.

## 7. Roles de agentes

### 7.1 Agente Solver (Backend) — dueño de `core/`
- **Hace:** formulación, ensamblaje, nuevos tipos de elemento/carga/apoyo, postproceso,
  catálogo de materiales y perfiles, `builders`, `design`.
- **No hace:** tocar `ui/` ni `app.py` (si un cambio de API lo exige, abre la tarea al
  Agente Frontend y deja la firma vieja funcionando hasta que se migre).
- **Entrega:** código + test analítico que falla antes y pasa después + nota de
  convención de signos si agrega magnitudes.
- **Criterio numérico:** el solver es exacto por barra (polinomios). Tolerancias de test
  `rel=1e-6`; si algo necesita tolerancias más laxas, justificarlo en el test.

### 7.2 Agente Frontend (UI/Visualización) — dueño de `ui/` y `app.py`
- **Hace:** inputs, layout, gráficos Plotly, textos, conversión de unidades de UI.
- **No hace:** ninguna cuenta estructural (ni "un M = qL²/8 rápido"): todo número
  mostrado sale de `Results`, `SafetyCheck` o `Section`.
- **Reglas:** `ui/plots.py` sin Streamlit (devuelve `go.Figure`); colores definidos como
  constantes al inicio del módulo; convención de M del lado traccionado por defecto.
- **Entrega:** app corriendo sin excepciones en ambos modos (Viga / Pórtico). Prueba
  mínima headless:
  ```python
  from streamlit.testing.v1 import AppTest
  at = AppTest.from_file("app.py").run(); assert not at.exception
  ```

### 7.3 Agente QA / Criterio Mecánico — dueño de `tests/` y del criterio de ingeniería
- **Hace:** casos analíticos (Roark, Timoshenko, Gere, tablas AISC/CIRSOC), verificación
  cruzada con `anastruct`, revisión de órdenes de magnitud, unidades y signos de cada PR.
- **Checklist de revisión de todo cambio en `core/`:**
  1. Equilibrio global: ΣF y ΣM de cargas + reacciones = 0.
  2. Signos: carga hacia abajo en viga simple ⇒ M > 0 en el tramo, δ < 0.
  3. Unidades: ninguna conversión ×1e3 / ×1e6 dentro de `core/`.
  4. Caso límite: barra única sin nodos intermedios da el mismo resultado que mallada.
  5. Inestabilidad: mecanismo ⇒ `StructuralError`, nunca resultados basura.
- **Cada test cita su fórmula** en el nombre o docstring (p. ej. `δ = 5qL⁴/384EI`).
- Puede bloquear un merge si falla el checklist, aunque los tests pasen.

## 8. Protocolo para agregar una funcionalidad sin romper el core

1. **Rama:** `feat/<rol>-<tema>` (p. ej. `feat/solver-internal-hinges`).
2. **Test primero (QA o Solver):** caso analítico en `tests/` que hoy falla, con la
   fórmula y su fuente.
3. **Core (Solver):** implementación mínima. Extender con parámetros por defecto o
   campos nuevos al final de las dataclasses; nunca cambiar firmas del §5.
4. **UI (Frontend):** exponer la función en `ui/inputs.py` / `ui/plots.py`.
5. **Checks:** `pytest && mypy && ruff check .` + prueba `AppTest`.
6. **Commit convencional:** `feat(core): …`, `fix(ui): …`, `test: …`, `docs: …`. Un tema por commit.
7. **Si cambia una convención** (unidades, signos, API): actualizar este archivo en el
   mismo commit. Un cambio de convención sin actualizar AGENTS.md se rechaza.

Ejemplo — agregar rótulas internas: (1) test viga Gerber con M = 0 en la rótula;
(2) campo `release_i/release_j: bool = False` al final de `Member`; condensación estática
en `local_stiffness`; (3) columna "rótula" en el editor de apoyos; (4) checks.

## 9. Alcance del MVP y backlog (fuera de alcance hoy)

Incluido: vigas continuas y voladizos, pórtico simple de una nave, apoyos empotrado /
articulado / móvil, cargas puntuales, momentos, distribuidas trapezoidales y parciales,
peso propio, catálogo de perfiles, N-V-M-σ-deformada, FS con alerta < FS mín.,
búsqueda del perfil más liviano.

Backlog priorizado (no implementar sin tarea explícita):
1. Rótulas internas y apoyos elásticos (resortes).
2. Combinaciones de carga (CIRSOC 301 / AISC 360 LRFD-ASD).
3. Pandeo flexional de columnas y pandeo lateral-torsional de vigas.
4. Tensión de corte (τ = VQ/It) y von Mises combinada.
5. Flexión en eje débil y secciones asimétricas (ya soportadas en `Section` vía `c_top/c_bot`).
6. Pórticos genéricos (editor de nodos/barras) y cargas térmicas.
7. Reporte PDF de memoria de cálculo.

## 10. Limitaciones que la UI debe seguir comunicando

Elástico lineal, pequeñas deformaciones, Euler-Bernoulli (sin deformación por corte),
sin efectos de segundo orden, sin pandeo ni fatiga, propiedades de catálogo nominales.
Es una herramienta de **predimensionamiento**, no reemplaza la memoria de cálculo.
