# AGENTS.md — Estructural 2D (MVP)

Guía operativa para agentes de IA (y humanos) que trabajan en este repositorio.
Leerla completa antes de tocar código. Si una instrucción de una tarea contradice
este archivo, gana este archivo salvo que la tarea diga explícitamente que lo modifica.

## 1. Qué es esto

App web (Streamlit) de **predimensionamiento rápido y cálculo elástico lineal 2D** de
vigas y pórticos planos para ingeniería mecánica/industrial. Método directo de rigidez,
elemento de pórtico Euler-Bernoulli, verificación por tensión equivalente de von Mises con
FS = Sy/σvM (σvM = σmax normal cuando gobierna la flexión; corte τ = VQ/It en el eje neutro).

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
app.py            Orquestación Streamlit. NO calcula nada. Exporta PNG de Plotly (kaleido) para el PDF.
 ├── ui/inputs.py   Sidebar -> Model. ÚNICO lugar con conversión de unidades de UI.
 ├── ui/plots.py    Model/Results -> go.Figure (sin Streamlit; reutilizable en reportes)
 └── core/          Motor puro. Sin Streamlit, sin Plotly, sin pandas, sin I/O.
      ├── materials.py     Material, Section, catálogo (IPE, IPN, UPN, W, C conformado, caños Sch40, tubos)
      ├── model.py         Model, Member, Support, Spring, NodalLoad, DistributedLoad (dataclasses)
      ├── builders.py      build_beam(), build_portal_frame(): definición "de ingeniero" -> Model
      ├── solver.py        solve(model) -> Results  (función pura)
      ├── verification.py  check_safety() -> SafetyCheck (FS, estado OK/ALERTA/FLUENCIA)
      ├── design.py        suggest_section(): perfil más liviano que verifica
      ├── glossary.py      Glosario y simbología: FUENTE ÚNICA para el PDF y la wiki (to_markdown())
      ├── reports.py       build_pdf_report() -> bytes: memoria A4 con figuras vectoriales (ReportLab)
      └── fonts/           DejaVu Sans (TTF + licencia) embebida en el PDF; no tocar sin motivo
tests/            pytest, casos analíticos con fórmula de referencia
```

Regla dura (se verifica en revisión):

| Módulo | Puede importar | NO puede importar |
|---|---|---|
| `core/*` | stdlib, `numpy`, otros `core/*` | `streamlit`, `plotly`, `pandas`, `ui/*` |
| `core/reports.py` | lo anterior + `reportlab` | ídem; y ningún otro módulo del core importa `reportlab` ni `core.reports` |
| `ui/plots.py` | `core`, `numpy`, `plotly` | `streamlit` |
| `ui/inputs.py`, `app.py` | todo | — |

Chequeo rápido: `grep -rnE "streamlit|plotly|pandas" core/` debe devolver vacío.
Ambas reglas están automatizadas en `tests/test_architecture_and_app.py`.
`core/__init__.py` **no** reexporta `build_pdf_report`: importar `core` no debe arrastrar ReportLab.
Excepción a "sin I/O": `core/reports.py` lee **sólo** las fuentes de `core/fonts/`, una vez al
importarse. Las figuras propias del PDF (esquema, deformada, diagramas N-V-M-σ) se dibujan con
`reportlab.graphics` desde `Model`/`Results`.
Exportación estática de figuras Plotly: permitida **sólo en la capa UI/App** y desacoplada del core.
`app.py` llama `fig.to_image(format="png")` (requiere `kaleido`) y pasa los bytes a
`build_pdf_report(..., image_bytes=...)`; si kaleido/Chrome no está disponible, la app captura la
excepción y genera el PDF sin esa imagen. `core/` nunca importa Plotly ni kaleido: sólo recibe
`bytes | None`.

## 4. Unidades y convenciones de signo

**Core: SI-mm sin excepciones** en datos, cálculos y valores devueltos.
Excepciones, sólo de presentación (texto, nunca números devueltos al resto del core):
- `core/reports.py` convierte a unidades de ingeniería **sólo para imprimir** (constantes `_M`, `_KN`,
  `_KNM`, `_CM*` al inicio del módulo).
- `core/verification.py` redacta `issues` en m, kN y kN·m (`_MM_TO_M_TEXT`, `_N_TO_KN_TEXT`,
  `_NMM_TO_KNM_TEXT`, helper `_where`).
- `Section.linear_mass(rho)` / `mass_per_m` devuelven kg/m (A [mm²]·1e-6·ρ [kg/m³]): es la fórmula de la
  masa lineal, no una conversión de UI; `mass_per_m` es la de catálogo en acero (ρ = 7850).
Notación: fluencia **S<sub>y</sub>** (`Sy`) en mensajes, PDF y glosario; `Fx`, `Fy` son fuerzas.

| Magnitud | Core | UI (sólo `ui/`) |
|---|---|---|
| Longitud | mm | m |
| Fuerza | N | kN |
| Carga distribuida | N/mm | kN/m (numéricamente igual) |
| Momento | N·mm | kN·m (×1e6) |
| Tensión, E | MPa (N/mm²) | MPa |
| Área / Inercia / W | mm² / mm⁴ / mm³ | cm² / cm⁴ / cm³ |
| Densidad | kg/m³ | kg/m³ |
| Rigidez de resorte kx, ky / krz | N/mm / N·mm/rad | kN/m (numéricamente igual) / kN·m/rad (×1e6) |

- Ejes globales: X → derecha, Y ↑ arriba, Mz antihorario +.
- Cargas en el core: `Fy < 0` y `q < 0` son **hacia abajo**. La UI muestra "+ hacia abajo" y
  convierte el signo en `ui/inputs.py`.
- Ejes locales de barra: x de nodo `i` a `j`, y local = x rotado +90°.
- Solicitaciones: **N > 0 tracción**; **M > 0 tracciona la fibra de y local negativa**
  (en vigas horizontales: fibra inferior, "sagging"); **V = dM/dx**.
- Tensiones: `sigma_top = N/A − M·c_top/I`, `sigma_bot = N/A + M·c_bot/I` (c_top ≠ c_bot en secciones
  asimétricas; con M > 0 la fibra superior se comprime). σ_max = máx(|σ_top|, |σ_bot|).
- Corte y von Mises (`core/verification.py`, `member_stresses`): τ = |V|·Q_max/(I·t_web) en el eje neutro
  [MPa]; σ_vM = máx(σ_max [fibra extrema, τ ≈ 0], √((N/A)² + 3τ²) [eje neutro]). FS = Sy/σ_vM y en ELU
  σ_vM,u ≤ φ·Sy. Con flexión dominante σ_vM = σ_max **exacto** (mismos FS, ubicación y textos de siempre).
  `Section` sin datos de corte (`Q_max = 0` o `t_web = 0`, p. ej. `custom` sin Q) ⇒ τ = 0. Si gobierna el
  corte, `issues` lo dice (`σvM = … (corte en el eje neutro, τ máx = …)`, en ELU `σvM,u`).
- Eje débil: `Section.weak_axis(flip=False)` devuelve la sección orientada para flexionar en su eje débil
  dentro del plano (I ↔ Iy, c_top/c_bot ↔ c_y1/c_y2, Q_max/t_web ↔ Q_y/t_y); el solver no cambia. En canales
  (UPN, C) c_y1 = e (dorso del alma) ≠ c_y2 = b − e (puntas); `flip=True` pone las puntas arriba. Momento
  fuera del plano: `weak_axis_bending_stress(section, M_y) = |M_y|/W_y` (no se combina con el del plano).
- Reacciones: en ejes globales, sobre la estructura.
- Rótulas internas: `Member.release_start` / `release_end` = M = 0 en el extremo i / j de esa
  barra (condensación estática local). El momento de un extremo liberado vale `0.0` exacto en
  `MemberResult.M` y `end_forces`. Un nudo donde **todas** las barras concurrentes tienen rótula
  no tiene rigidez a giro: su `rz` se informa 0 (cada barra gira por su cuenta) y un `Mz` aplicado
  ahí lanza `StructuralError` (salvo que el nudo tenga resorte rotacional `krz > 0`).
- Apoyos elásticos: `Spring(node, kx, ky, krz)` en `Model.springs` (ejes globales, k ≥ 0 finita, un
  `Spring` por nodo). Se suman a la diagonal de K; un GDL con resorte y sin vínculo rígido es libre.
  Reacción del resorte sobre la estructura **R = −k·u** (opuesta al desplazamiento/giro), incluida en
  `Results.reactions[node]` junto con la del `Support` del mismo nudo si lo hay. `Results.reactions`
  trae también los nodos que sólo tienen resorte. Un modelo puede apoyarse sólo en resortes.
- `SupportType.FREE` (`"libre"`) no restringe ningún GDL (`restrained_dofs = (False, False, False)`):
  sirve para que la UI declare un apoyo puramente elástico. No se dibuja símbolo rígido; libre sin
  resorte ⇒ mecanismo (`StructuralError`).
- Presentación de resortes (UI y PDF): Kx, Ky en kN/m (= N/mm) y Krz en kN·m/rad; rótulo del vínculo
  `"<rígido> + elástico"`, o `"elástico"` si el nodo es libre o sólo tiene resorte; la tabla de
  reacciones (app y PDF) lista también los nodos que sólo tienen resorte. Kx se dibuja hacia afuera
  de la estructura.
- Resortes inertes: un `k` sobre un GDL que el `Support` del nudo ya restringe **no trabaja** (u = 0):
  `Spring.inert(type)`
  lo detecta y `Spring.active(type)` lo anula. La UI avisa y lo descarta; `ui/plots.py` y el PDF dibujan
  sólo resortes activos (Kx hacia afuera de la estructura; con apoyo rígido, Ky corrido y Krz como arco
  superior) y la tabla 2.2 informa el inerte como "sin efecto".
- Dibujo de hipótesis (UI y PDF, misma convención): D naranja, L violeta; en una barra con D y L del
  mismo sentido la L se apila sobre la D y el rótulo de la D pasa al otro lado de la barra; con cargas L
  los rótulos indican la hipótesis (`qD`, `qL`, `(D)`, `(L)`). Una entrada de leyenda por hipótesis.
- Hipótesis de carga: `NodalLoad.case` / `DistributedLoad.case` (y `BeamPointLoad` / `BeamDistLoad` de
  `builders`) = `LoadCase.DEAD` (`"D"`, **por defecto**: el peso propio y toda carga sin `case` es D) o
  `LoadCase.LIVE` (`"L"`). `solve()` resuelve **todas** las cargas con γ = 1 (sin L es la resolución de
  siempre). Las combinaciones (`LoadCombination(name, gamma_D, gamma_L, limit_state)`, `"ELU"`/`"ELS"`)
  se resuelven por superposición: K se ensambla una vez, se resuelven D y L y se combinan con
  `combine_results` (exacto: `MemberResult.polynomials` guarda N, V, M, u, v por barra y se re-muestrea con
  la misma regla de estaciones que `solve`). Estándar (CIRSOC 301-2005 §A.4.2 / AISC 360-16 §B2):
  `COMB_1_4D`, `COMB_1_2D_1_6L` (ELU) y `COMB_SERVICE_D_L` = 1.0D + 1.0L (ELS).
- Verificación por combinaciones (`check_combinations`): ELU σ_u ≤ φ·Sy con φ = 0.90 (AISC 360-16 §F1),
  tensión elástica de primera fluencia; gobierna la combinación de máxima σ_u/(φ·Sy) (= máximo de la
  envolvente). ELS: flecha bajo 1.0D + 1.0L **sin mayorar** contra L/N.
- Veredicto global por combinaciones (`CombinationCheck.verdict` → `(Status, rótulo)`, fuente única para
  UI y PDF): `FLUENCIA` (σ_u > Sy) y `NO VERIFICA` (φ·Sy < σ_u ≤ Sy) en rojo (`Status.FAIL`); `ALERTA` en
  amarillo sólo si ELU verifica y falla la flecha ELS; `VERIFICA` en verde. La app y la búsqueda de perfil
  (`suggest_section(..., phi=...)`) usan este mismo criterio; el PDF documenta ELU con la combinación
  gobernante (cargas mayoradas) y la deformación (§3.3) sólo con 1.0D + 1.0L.
- `Results.extreme`: máximo de |valor|; magnitudes que difieren < 1e-12 (relativo, `_TIE_RTOL`) empatan y
  gana la 1ª estación de la 1ª barra (resultado independiente del redondeo de la plataforma).

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
core.verification.check_safety(model, results, fs_min=1.5, deflection_limit_ratio=300.0,
                               reference_length=None) -> SafetyCheck
core.verification.SafetyCheck: .status, .ok, .strength_ok, .issues; ELS: .delta_max, .deflection_ratio,
                               .deflection_limit_ratio, .deflection_ok, .reference_length, .delta_adm
core.builders.build_beam(...) / build_portal_frame(..., beam_q_case=LoadCase.DEAD,
                                                    lateral_case=LoadCase.DEAD) -> Model
core.model.Member(i, j, section, material, release_start=False, release_end=False)  # rótulas
core.model.Spring(node, kx=0.0, ky=0.0, krz=0.0); Model.springs: list[Spring] = []  # resortes
core.model.Spring.inert(support_type) -> tuple[str, ...]; .active(support_type) -> Spring; .is_null
core.model.LoadCase.DEAD | LoadCase.LIVE  # "D" | "L"; campo `case=LoadCase.DEAD` al final de NodalLoad,
                                          # DistributedLoad, BeamPointLoad, BeamDistLoad
core.model.LoadCombination(name, gamma_D, gamma_L, limit_state="ELU")  # "ELU" | "ELS"
core.model.COMB_1_4D, COMB_1_2D_1_6L, COMB_SERVICE_D_L, ULS_COMBINATIONS; Model.load_cases(), .has_live_loads()
core.solver.solve_load_cases(model, n_stations=41) -> dict[LoadCase, Results]  # D siempre; L si hay
core.solver.combine_results(res_D, res_L, gamma_D, gamma_L) -> Results         # superposición exacta
core.solver.solve_combinations(model, combinations, n_stations=41) -> list[Results]  # K una sola vez
core.solver.combine_cases(cases, combinations) -> list[Results]  # combina salidas de solve_load_cases
core.solver.MemberResult.polynomials: MemberPolynomials | None                 # solución exacta por barra
core.verification.check_combinations(model, phi=0.9, deflection_limit_ratio=300.0, reference_length=None,
                                     uls_combinations=ULS_COMBINATIONS,
                                     service_combination=COMB_SERVICE_D_L, n_stations=41,
                                     case_results=None) -> CombinationCheck  # case_results: no re-resuelve
core.verification.CombinationCheck: .governing, .strength, .utilization, .strength_checks, .status, .ok,
                                    .issues, .verdict -> (Status, "VERIFICA"|"ALERTA"|"NO VERIFICA"|"FLUENCIA");
                                    ELS: .service_combination, .service_check, .delta_max,
                                    .deflection_ok; .governing_results, .service_results,
                                    .uls_results (una por combinación ELU, orden de strength_checks)
core.design.suggest_section(build, family, fs_min, max_deflection=None,
                            deflection_limit_ratio=300.0, phi=None) -> Suggestion | None
                            # phi dado: verifica con check_combinations (fs_min se ignora); la app usa φ = 0.90
core.design.Suggestion(section, check, combination=None)  # combination: CombinationCheck si se usó phi
core.materials.SECTIONS[family][name] -> Section             # claves de familia estables (ver abajo)
core.materials.Section(..., Q_max=0, t_web=0, Iy=0, Wy=0, c_y1=0, c_y2=0, Q_y=0, t_y=0)  # al final [mm³,mm,mm⁴]
core.materials.Section.weak_axis(flip=False) -> Section; .W_top, .W_bot, .c_y_max, .has_shear_data,
                                                         .shear_factor  # Q/(I·t) [1/mm²]
core.materials.custom(name, A, I, c, *, Q_max=0.0, t_web=0.0, Iy=0.0, Wy=0.0)
core.verification.shear_stress(section, V) / von_mises(sigma, tau) -> FloatArray  # [MPa]
core.verification.member_stresses(section, member_result) -> MemberStresses  # σ_top/bot, τ, σ_vM por estación
core.verification.weak_axis_bending_stress(section, M_y) -> float  # |M_y| / W_y [MPa]
core.verification.SafetyCheck: .tau_max, .sigma_vm_max, .shear_governs  # de la barra crítica;
                               # .sigma_max sigue siendo la tensión NORMAL máx; .fs y .utilization usan σ_vM
core.verification.CombinationCheck: .tau_max, .sigma_vm_max   # de la combinación gobernante
# ---- Estabilidad (backlog 3, AISC 360-16 = CIRSOC 301-2005) ----
core.materials.Section(..., Z=0, Z_y=0, J=0, Cw=0, h0=0)   # al final [mm³, mm³, mm⁴, mm⁶, mm]; Z = 2·Q auto
core.materials.Section.r_x, .r_y, .r_min, .r_ts [mm]; .has_ltb_data   # r_x: eje de flexión (I)
core.materials.custom(..., Z=0.0, J=0.0, Cw=0.0, h0=0.0); CLOSED_FAMILIES; SOLID_FAMILY
core.verification.column_buckling_capacity(section, L, K=1.0, Fy=250.0, phi_c=0.9, *, E=200000.0,
                                           N_comp=0.0) -> ColumnBucklingCheck  # .Pn, .phi_Pn, .utilization
core.verification.beam_ltb_capacity(section, Lb, Fy=250.0, Cb=1.0, phi_b=0.9, *, E=200000.0,
                                    M_u=0.0) -> BeamLTBCheck  # .Mp, .Lp, .Lr, .Mn, .phi_Mn, .zone, .utilization
core.verification.interaction_h1(Pr, Pc, Mr, Mc) -> float     # AISC H1-1a/b
core.verification.physical_members(model) -> tuple[tuple[int, ...], ...]  # cadenas colineales
core.verification.check_safety(..., *, stability=False, unbraced_length=None, effective_length_factor_k=1.0)
core.verification.check_combinations(..., *, stability=False, unbraced_length=None,
                                     effective_length_factor_k=1.0)   # estabilidad sólo en ELU
core.verification.SafetyCheck.stability: tuple[MemberStabilityCheck, ...] = ()  # una por barra física
core.verification.SafetyCheck: .critical_stability, .stability_utilization, .stability_ok, .buckling_failure,
                               .stability_governs, .uls_ok, .stability_notes
core.verification.CombinationCheck: .stability_governing -> (comb, SafetyCheck), .stability_utilization,
                                    .stability_ok, .stability_governs
core.materials.Section.linear_mass(rho) -> float [kg/m]       # mass_per_m = linear_mass(7850), acero
core.materials.FAMILY_NOTES[family] -> str                   # fuente y supuestos de cada familia
core.materials.family_warnings(family) -> tuple[str, ...]    # advertencias de alcance (UI y PDF)
core.reports.build_pdf_report(model, results, check: SafetyCheck | CombinationCheck,
                              project_title="Memoria de Cálculo", author="",
                              *, date=None, reference_length=None, include_figures=True,
                              include_glossary=True, image_bytes: bytes | None = None) -> bytes  # b"%PDF..."
    # check = CombinationCheck: §3.1-3.2 de check.governing_results, §3.3 de check.service_results,
    # §4 de check.verdict / check.issues (`results` no se usa). image_bytes: PNG exportado por la UI.
core.glossary.NOTATION / GRAPHIC_SYMBOLS / TERMS             # datos (mini-notación x_{sub}, x^{sup})
core.glossary.to_markdown(include_graphics=True, heading_level=2) -> str  # para st.markdown (LaTeX $..$)
ui.plots.plot_structure(model, results=None, deformed_scale=None, ...) -> go.Figure
ui.plots.plot_diagram(model, results, quantity, moment_on_tension_side=True) -> go.Figure
```

Claves de familia de `SECTIONS` (estables; renombrarlas es un cambio de API): `"IPE"`, `"IPN"`,
`"UPN"`, `"Perfil W"`, `"Perfil C (Conformado)"`, `"Tubo circular"`, `"Tubo cuadrado"`,
`"Tubo rectangular"`. Fuera del catálogo, los constructores usan `"Macizo"` y `"Personalizado"`.
Toda familia nueva lleva entrada en `FAMILY_NOTES` (hay test que lo controla).

Verificación (`Status`): `FLUENCIA` si FS < 1; `ALERTA` si FS < FS_min **o** la flecha supera
L/N (estado límite de servicio, sin fluencia); `OK` sólo si verifican resistencia y flecha.
Con `stability=True`: `FLUENCIA` (severidad) también si una barra física es inestable con resistencias
nominales (utilización H1 nominal > 1; `issues` dice "Inestabilidad…" y el PDF "INESTABILIDAD"); `ALERTA` si
la utilización H1 con Pn/FS_min, Mn/FS_min supera 1. En combinaciones, Pu/Mu frente a φPn/φMn: no verificar
da `verdict = (FAIL, "NO VERIFICA")`. `strength_ok` sigue siendo **sólo** tensiones de la sección;
`stability_ok` / `uls_ok` incluyen la estabilidad. Sin `stability=True` el resultado es idéntico al anterior.
`deflection_limit_ratio = 0` desactiva el control de flecha. L por defecto: ancho horizontal del
modelo (`default_reference_length`). `max_deflection` [mm] de `suggest_section`, si se da,
reemplaza a `deflection_limit_ratio`. El motivo de un veredicto no-OK se lee de
`SafetyCheck.issues` (castellano, listo para mostrar); la UI no lo reconstruye.

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
  en la UI (`st.cache_data`): `app.py` cachea análisis, PNG, PDF y sugerencia con clave `repr(model)`
  (+ parámetros), y resuelve D/L una sola vez por cambio de modelo (`check_combinations(case_results=…)`).
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
  Las notas y advertencias de una familia de perfiles se leen de `core.materials.FAMILY_NOTES`
  y `family_warnings()`; la UI no redacta criterio de ingeniería propio.
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
  3. Unidades: ninguna conversión ×1e3 / ×1e6 dentro de `core/` (salvo la de impresión de
     `core/reports.py`, §4).
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
(2) campo `release_start/release_end: bool = False` al final de `Member`; condensación estática
en `local_stiffness`; (3) columna "rótula" en el editor de apoyos; (4) checks.

## 9. Alcance del MVP y backlog (fuera de alcance hoy)

Incluido: vigas continuas y voladizos, pórtico simple de una nave, apoyos empotrado /
articulado / móvil, cargas puntuales, momentos, distribuidas trapezoidales y parciales,
peso propio, catálogo de perfiles, N-V-M-σ-deformada, FS con alerta < FS mín.,
búsqueda del perfil más liviano.

Backlog priorizado (no implementar sin tarea explícita):
1. ~~Rótulas internas y apoyos elásticos~~ → **completado**: core (`Member.release_start/release_end`,
   `Spring`, `Model.springs`), UI (editor de rótulas con una fila fija por barra, resortes Kx/Ky/Krz y apoyo
   `libre`), símbolos en `ui/plots.py` y PDF (glosario en `core/glossary.py`), tabla de reacciones con
   nodos elásticos. Tests: `test_hinges.py`, `test_springs.py`, `test_reports.py`, `test_plots.py`,
   `test_architecture_and_app.py`.
2. ~~Combinaciones de carga D / L (CIRSOC 301 / AISC 360 LRFD)~~ → **completado**: core (`LoadCase`,
   `LoadCombination`, `solve_load_cases`, `combine_results`, `solve_combinations`, `check_combinations`,
   `CombinationCheck.verdict`), UI (hipótesis D/L por carga, veredicto global, diagramas por combinación,
   búsqueda de perfil por combinaciones) y PDF (LRFD + ELS de servicio). Tests: `test_load_combinations.py`,
   `test_reports.py`, `test_deflection.py`. q y H del pórtico eligen hipótesis D o L
   (`beam_q_case`, `lateral_case`). Fuera de alcance (tarea aparte): ASD y cargas Lr / S / R / W / E
   (H lateral no se trata como viento: entra como D o L).
3. ~~Pandeo flexional de columnas y pandeo lateral-torsional de vigas~~ → **core completado**:
   `Section.Z/Z_y/J/Cw/h0`, `r_x/r_y/r_min/r_ts` (IPE/W: J El Darwish-Johnston con acuerdos y Cw = Iy·h0²/4,
   < 1 % de tabla; IPN: J sin acuerdos, conservador; UPN/C: pared delgada DG9), `column_buckling_capacity`
   (E3), `beam_ltb_capacity` (F2 doble T y canales, F11 planchuelas; sin PLT en eje débil, tubos, redondos),
   `interaction_h1`, `physical_members` y `check_safety/check_combinations(stability=True, ...)`.
   Test: `test_buckling.py`. Criterio: KL y Lb sobre la **barra física** (los constructores parten vigas en
   cada carga puntual); Cb = 1; Lb = mín(unbraced_length, L). **Pendiente Frontend/PDF**: checkbox
   "verificar estabilidad" + inputs Lb [m] y K en `ui/inputs.py`, tabla por barra física (KL/r, φPn, Lb,
   zona, φMn, utilización H1) en `app.py` y §4 del PDF, `stability_notes`, y símbolos (λ, Fe, Fcr, Pn, K, Lb,
   Lp, Lr, Mp, Mn, Cb, J, Cw, r_ts) en `core/glossary.py`. Pasar `stability` a `suggest_section`.
4. ~~Tensión de corte (τ = VQ/It) y von Mises combinada~~ → **core completado**: `Section.Q_max/t_web`
   para todo el catálogo (IPE/W por geometría con acuerdos deducidos del área de tabla; IPN/UPN con tf
   medio; tubos, macizos y C conformado exactos), `shear_stress`, `von_mises`, `member_stresses`,
   `SafetyCheck.tau_max/sigma_vm_max`, FS y ELU sobre σ_vM. Test: `test_shear_von_mises.py`.
   **Pendiente Frontend/PDF**: mostrar τ_máx y σ_vM (métrica junto a "σ máx" en `app.py`, diagrama τ en
   `ui/plots.py`, fila en la tabla de verificación de `core/reports.py`) y sus símbolos en `core/glossary.py`.
5. ~~Flexión en eje débil y secciones asimétricas~~ → **core completado**: `Section.Iy/Wy/c_y1/c_y2/Q_y/t_y`
   (IPE/W geometría; IPN/UPN Iy y e de tabla; C conformado exacto), `Section.weak_axis()`,
   `weak_axis_bending_stress`, `W_top/W_bot`. `suggest_section(lambda s: build(s.weak_axis()), ...)` ya
   busca el perfil en eje débil. **Pendiente Frontend**: selector "eje fuerte / eje débil (invertido)" en
   `ui/inputs.py`. Fuera de alcance: flexión esviada combinada (M_z + M_y en una misma sección).
6. Pórticos genéricos (editor de nodos/barras) y cargas térmicas.
7. ~~Reporte PDF de memoria de cálculo~~ → hecho (`core/reports.py`), con esquema de cargas,
   deformada, diagramas N-V-M-σ vectoriales, convenciones de signo y glosario/simbología (§6 del PDF).
   Si se cambia un gráfico de `ui/plots.py`, revisar que el equivalente del PDF siga la misma
   convención (colores, M del lado traccionado). Todo símbolo, ícono o término nuevo que aparezca
   en la memoria o en la app se agrega **sólo** a `core/glossary.py`: el PDF (sección 6) y la wiki
   de la app lo leen de ahí; ninguno define glosario propio (hay test que lo controla).

## 10. Limitaciones que la UI debe seguir comunicando

Elástico lineal, pequeñas deformaciones, Euler-Bernoulli (sin deformación por corte),
sin efectos de segundo orden, sin fatiga, propiedades de catálogo nominales.
Estabilidad (sólo con `stability=True`): solicitaciones de primer orden (sin B1/B2 ni análisis directo; en
pórticos desplazables el K debe representar el desplazamiento lateral), Cb = 1, KL con la barra física y r_min
(no aprovecha arriostramientos intermedios), sin pandeo local (§E7), torsional ni flexotorsional (§E4);
fórmulas de acero (orientativas para aluminio). En canales UPN el Cw de pared delgada con tf medio es ≈ 15 %
mayor que el de tabla. Secciones personalizadas sin J/Cw/h0: PLT no evaluado (`stability_notes`).
von Mises se evalúa sólo en la fibra extrema y en el eje neutro (no en la unión ala-alma de los
doble T, donde σ y τ coexisten); τ es la de Jourawski (corte elástico, sin plastificación del alma).
Flecha: δ_máx es el desplazamiento total máximo de toda la estructura (en pórticos incluye el
desplazamiento lateral) referido a una única L; en voladizos L es la longitud del voladizo.
Por familia (`family_warnings()`): torsión no modelada en canales UPN / C cuando la carga no
pasa por el centro de corte; abolladura local no verificada en chapa conformada en frío.
Es una herramienta de **predimensionamiento**, no reemplaza la memoria de cálculo.
