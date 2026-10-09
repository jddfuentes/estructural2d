# Wiki técnica de ingeniería

> Estado: documentación del MVP. Esta página registra lo que el programa hace hoy,
> sus hipótesis y los puntos que deben validarse antes de usarlo como memoria de
> cálculo definitiva.

## 1. Alcance actual

Estructural 2D resuelve vigas y pórticos planos mediante el método directo de
rigidez. El modelo es lineal-elástico, de pequeñas deformaciones y con elementos
de pórtico Euler-Bernoulli. El resultado es apropiado para predimensionamiento y
para revisar órdenes de magnitud; no reemplaza una memoria firmada ni la
verificación normativa completa.

El núcleo trabaja exclusivamente en unidades SI-mm:

| Magnitud | Unidad |
| --- | --- |
| Longitud | mm |
| Fuerza | N |
| Carga distribuida | N/mm |
| Momento | N·mm |
| Tensión y módulo elástico | MPa = N/mm² |
| Área, inercia y módulo resistente | mm², mm⁴, mm³ |

La interfaz convierte y muestra m, kN, kN/m y kN·m. Esas conversiones no forman
parte de las ecuaciones del núcleo.

## 2. Sistema de referencia y signos

El sistema global es dextrógiro y plano:

- `+X`: hacia la derecha.
- `+Y`: hacia arriba.
- `+Z`: sale del plano, asociado al giro `rz`.
- `+Mz`: antihorario.
- `Fx > 0`: hacia `+X`.
- `Fy > 0`: hacia `+Y`; por eso una carga hacia abajo tiene `Fy < 0`.
- `q < 0` en dirección global Y: carga distribuida hacia abajo.
- `N > 0`: tracción.
- `M > 0`: tracciona la fibra de `y` local negativa. En una viga horizontal,
  corresponde a la fibra inferior y al momento sagante.
- `V = dM/dx`.

Cada barra tiene un eje local `x` que va del nodo `i` al nodo `j` y un eje local
`y` girado 90° en sentido antihorario. Las cargas pueden estar expresadas en
los ejes globales o en el eje local transversal.

## 3. Modelo cinemático

Cada nodo tiene tres grados de libertad:

$$
\mathbf{u}_i = [u_{x,i},\; u_{y,i},\; \theta_{z,i}]^T
$$

La hipótesis Euler-Bernoulli supone que las secciones permanecen planas y
normales al eje deformado. Por tanto, no se incluye deformación por corte y la
rotación de la sección se obtiene de la pendiente de la elástica. Esta hipótesis
es razonable para barras esbeltas; puede subestimar la flecha en barras cortas,
profundas o muy flexibles al corte.

Las propiedades de cada barra son constantes a lo largo de ella: `E`, `A` e
`I`. El material se modela como isótropo y lineal-elástico; no se consideran
plasticidad, fisuración ni degradación de rigidez.

## 4. Rigidez del elemento de pórtico

Para una barra de longitud `L`, en coordenadas locales, se utiliza la matriz
clásica de rigidez axial-flexional:

$$
\mathbf{k}_{local} =
\begin{bmatrix}
EA/L & 0 & 0 & -EA/L & 0 & 0 \\
0 & 12EI/L^3 & 6EI/L^2 & 0 & -12EI/L^3 & 6EI/L^2 \\
0 & 6EI/L^2 & 4EI/L & 0 & -6EI/L^2 & 2EI/L \\
-EA/L & 0 & 0 & EA/L & 0 & 0 \\
0 & -12EI/L^3 & -6EI/L^2 & 0 & 12EI/L^3 & -6EI/L^2 \\
0 & 6EI/L^2 & 2EI/L & 0 & -6EI/L^2 & 4EI/L
\end{bmatrix}
$$

El vector de grados de libertad local es

$$
[u_i,\; v_i,\; \theta_i,\; u_j,\; v_j,\; \theta_j]^T
$$

La matriz de transformación usa `c = cos(theta)` y `s = sin(theta)` para pasar
de desplazamientos globales a locales. La rigidez se ensambla en global como:

$$
\mathbf{K}_e = \mathbf{T}^T \mathbf{k}_{local} \mathbf{T}
$$

### Criterio y motivo

Se usa esta formulación porque representa en un único elemento la extensión
axial y la flexión plana, permite barras inclinadas y es exacta para la rigidez
elástica del elemento bajo las hipótesis adoptadas. No se agregan grados de
libertad de corte, torsión ni flexión fuera del plano porque el alcance actual es
un pórtico plano de primer orden.

## 5. Cargas distribuidas

Una carga distribuida lineal se define por sus valores en los extremos `q1` y
`q2`, por lo que puede ser uniforme o trapezoidal:

$$
q(x) = q_1 + (q_2-q_1)\frac{x}{L}
$$

Primero se proyecta la carga global sobre los ejes locales de la barra. Luego se
calcula el vector nodal equivalente con las funciones cúbicas de forma de
Euler-Bernoulli:

$$
\mathbf{f}_{eq} = \int_0^L \mathbf{N}(x)^T\mathbf{q}(x)\,dx
$$

El código evalúa esta integral con cuadratura de Gauss. Las cargas puntuales y
los momentos nodales se incorporan directamente al vector global `F`.

El peso propio, cuando se activa, se aproxima como una carga uniforme:

$$
q_g = -\rho A g
$$

con la conversión correspondiente a N/mm dentro del constructor del modelo.
Su signo negativo representa la dirección global `-Y`.

## 6. Resolución y reacciones

Después de ensamblar todas las barras y cargas, el sistema lineal es:

$$
\mathbf{K}\mathbf{U} = \mathbf{F}
$$

Los apoyos eliminan los grados de libertad restringidos. El solver escala el
sub-sistema libre antes de resolverlo y controla su condición numérica para
rechazar mecanismos o modelos inestables en lugar de devolver resultados sin
sentido.

Las reacciones se recuperan del desequilibrio global:

$$
\mathbf{R} = \mathbf{K}\mathbf{U} - \mathbf{F}
$$

La salida contiene `Rx`, `Ry` y `Mz` en cada nodo con apoyo. Una validación
fundamental para cada caso es el equilibrio global de fuerzas y momentos.

## 7. Esfuerzos internos y deformada

Para cada barra se transforma el desplazamiento global a local y se recuperan
las fuerzas de extremo. Con la carga lineal interpolada se construyen los
polinomios exactos de la barra:

$$
\frac{dN}{dx} = -q_x,\qquad
\frac{dV}{dx} = q_y,\qquad
\frac{dM}{dx} = V
$$

La deformada se obtiene integrando:

$$
\frac{du}{dx} = \frac{N}{EA},\qquad
\frac{d^2v}{dx^2} = \frac{M}{EI}
$$

Se agregan estaciones en los extremos, en los ceros de `V` y en los puntos donde
la pendiente de la elástica se anula. Esto permite capturar máximos internos de
momento y desplazamiento sin depender únicamente de una malla visual.

## 8. Tensión normal

La tensión longitudinal se calcula mediante superposición de esfuerzo axial y
flexión:

$$
\sigma_{top} = \frac{N}{A} - \frac{M c_{top}}{I}
$$

$$
\sigma_{bot} = \frac{N}{A} + \frac{M c_{bot}}{I}
$$

La magnitud reportada es el máximo de `|sigma_top|` y `|sigma_bot|` en todas las
estaciones. El criterio actual sólo verifica tensión normal. No incluye tensión
de corte, interacción axial-flexión mediante von Mises, concentraciones,
abolladura local ni plastificación.

## 9. Verificación de seguridad del MVP

Para cada barra se calcula el factor de seguridad elástico frente a fluencia:

$$
FS = \frac{S_y}{|\sigma|_{max}}
$$

El valor de la estructura es el menor `FS` de todas las barras. El umbral de UI
por defecto es `FS_min = 1.5`:

| Estado | Criterio |
| --- | --- |
| `OK` | `FS >= FS_min` |
| `ALERTA` | `1 <= FS < FS_min` |
| `FLUENCIA` | `FS < 1` |

Este no es todavía un criterio normativo completo. El factor debe revisarse según
el material, la norma aplicable, las combinaciones de carga y si se trabaja con
un enfoque admisible o resistente.

## 10. Supuestos y exclusiones explícitas

Actualmente se supone:

- comportamiento lineal-elástico y pequeñas deformaciones;
- geometría y propiedades constantes por barra;
- uniones rígidas entre barras, salvo los apoyos definidos;
- elemento Euler-Bernoulli sin deformación por corte;
- cargas estáticas aplicadas en nodos o distribuidas linealmente sobre barras;
- análisis de primer orden, sin efectos P-Delta;
- flexión en un solo eje y sin torsión fuera del modelo plano;
- propiedades nominales del catálogo, sin tolerancias de fabricación;
- tensiones normales evaluadas en fibras extremas elásticas.

Fuera de alcance actual: pandeo flexional, pandeo lateral-torsional, abolladura
local, corte, torsión, fatiga, vibraciones, contacto, grandes deformaciones,
plasticidad, combinaciones normativas y diseño de uniones.

## 11. Registro de validación técnica

Cada ampliación debería agregar un caso analítico con fórmula, unidades,
convención de signos y tolerancia. El estado de referencia del MVP es:

| Área | Evidencia actual | Próxima validación recomendada |
| --- | --- | --- |
| Viga simplemente apoyada | Fórmulas clásicas de `Mmax`, `Vmax` y `delta` | Revisar casos de carga parcial y trapezoidal |
| Voladizo | Reacciones, momento y flecha analíticos | Casos con `Fx` y momento nodal |
| Pórtico plano | Casos analíticos básicos y equilibrio | Comparación independiente con software FEM |
| Rigidez de sección | Catálogo y geometría nominal | Contrastar propiedades con proveedor |
| Seguridad | `FS = Sy / sigma_max` | Definir norma, combinaciones y criterio de diseño |
| Peso propio | `q = -rho A g` | Revisar masas, unidades y ejes para perfiles reales |

### Pendientes de decisión

1. Definir norma de diseño objetivo (por ejemplo, reglamento local o AISC) y
   distinguir estados límite de servicio y resistencia.
2. Definir combinaciones de carga y factores de mayoración.
3. Incorporar corte, interacción de tensiones y pandeo cuando el alcance lo
   requiera.
4. Registrar para cada perfil la fuente exacta de `A`, `I`, `c`, masa y límites de
   validez.
5. Mantener casos de regresión con equilibrio global y comparación independiente.

## 12. Registro de cambios

- **MVP actual:** método directo de rigidez 2D, elemento Euler-Bernoulli,
  postproceso N-V-M, deformada, tensión normal y verificación elástica.
- Las decisiones nuevas deben agregarse aquí indicando fecha, motivación, impacto
  en signos/unidades y tests de validación asociados.
