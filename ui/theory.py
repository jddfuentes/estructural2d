"""Bases teóricas y criterios de cálculo mostrados en la interfaz."""

from __future__ import annotations

import streamlit as st


def render_theory() -> None:
    """Renderiza las bases teóricas del análisis de pórticos 2D."""
    st.markdown(
        r"""
## Bases Teóricas y Criterios de Cálculo

### Introducción y alcance

Estructural 2D utiliza el **Método Directo de Rigidez** para el análisis elástico
lineal de primer orden de vigas y pórticos planos. El modelo supone pequeñas
deformaciones, propiedades constantes por barra y comportamiento estático. Los
resultados son adecuados para predimensionamiento: no sustituyen la verificación
reglamentaria completa de pandeo, segundo orden, fatiga, corte ni efectos locales.

### Hipótesis fundamentales

- **Material:** homogéneo, isótropo y elástico lineal. La relación constitutiva
  uniaxial sigue la ley de Hooke, $\sigma = E\,\varepsilon$.
- **Euler-Bernoulli:** las secciones planas permanecen planas y ortogonales al eje
  neutro después de deformarse. Se desprecia la distorsión por corte, hipótesis
  apropiada como aproximación de ingeniería para $L/h > 5$.
- **Superposición:** al ser lineales las relaciones entre cargas, desplazamientos y
  esfuerzos, la respuesta total es la suma de las respuestas individuales.

### Elemento de pórtico 2D

Cada nodo tiene dos traslaciones y una rotación. Para el nodo $k$, el vector de
desplazamientos nodales es

$$
\mathbf{u}_k =
\begin{bmatrix}
u_k & v_k & \theta_k
\end{bmatrix}^{T}.
$$

Para una barra de longitud $L$, módulo elástico $E$, área $A$ e inercia $I$, la
relación local entre fuerzas y desplazamientos es
$\mathbf{f}_e = \mathbf{k}_e\,\mathbf{u}_e$, con
$\mathbf{u}_e = [u_i,v_i,\theta_i,u_j,v_j,\theta_j]^T$ y

$$
\mathbf{k}_e =
\begin{bmatrix}
\frac{EA}{L} & 0 & 0 & -\frac{EA}{L} & 0 & 0 \\
0 & \frac{12EI}{L^3} & \frac{6EI}{L^2} & 0 & -\frac{12EI}{L^3} & \frac{6EI}{L^2} \\
0 & \frac{6EI}{L^2} & \frac{4EI}{L} & 0 & -\frac{6EI}{L^2} & \frac{2EI}{L} \\
-\frac{EA}{L} & 0 & 0 & \frac{EA}{L} & 0 & 0 \\
0 & -\frac{12EI}{L^3} & -\frac{6EI}{L^2} & 0 & \frac{12EI}{L^3} & -\frac{6EI}{L^2} \\
0 & \frac{6EI}{L^2} & \frac{2EI}{L} & 0 & -\frac{6EI}{L^2} & \frac{4EI}{L}
\end{bmatrix}.
$$

Los términos $EA/L$ representan rigidez axial; los términos con $EI$ representan
rigidez flexional.

### Transformación global y ensamblaje

Si $\alpha$ es el ángulo de la barra respecto del eje global $X$, se define
$c=\cos\alpha$ y $s=\sin\alpha$. La matriz de transformación es

$$
\mathbf{T} =
\begin{bmatrix}
c & s & 0 & 0 & 0 & 0 \\
-s & c & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & c & s & 0 \\
0 & 0 & 0 & -s & c & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}.
$$

La rigidez de la barra expresada en coordenadas globales es

$$
\mathbf{K}_e = \mathbf{T}^{T}\mathbf{k}_e\mathbf{T}.
$$

Luego, las contribuciones de todos los elementos se ensamblan en los grados de
libertad globales y se resuelve

$$
\mathbf{K}_{global}\,\mathbf{U}
= \mathbf{F}_{nodal} + \mathbf{F}_{eq}.
$$

Las cargas distribuidas se convierten en fuerzas nodales equivalentes mediante
las funciones de forma cúbicas de Hermite, que interpolan la flexión y el giro a
lo largo de cada barra. Para cargas no uniformes o parciales, las integrales de
esas funciones se evalúan mediante cuadratura de Gauss-Legendre. El vector
$\mathbf{F}_{eq}$ conserva el trabajo virtual de la carga distribuida sobre los
desplazamientos compatibles del elemento.

### Esfuerzos internos y deformada

Una vez obtenidos los desplazamientos, se recuperan las solicitaciones de cada
barra en forma continua. La fuerza normal $N(x)$, el corte

$$
V(x) = \frac{dM(x)}{dx},
$$

y el momento flector $M(x)$ se evalúan a lo largo de la longitud, respetando los
signos de los ejes locales. La deformada flexional se obtiene de

$$
EI\,\frac{d^2v(x)}{dx^2}=M(x).
$$

La integración continua de las expresiones polinómicas de cada barra, con las
condiciones de borde de sus nodos, produce la elástica exacta dentro del modelo
Euler-Bernoulli; no se aproxima con una sucesión de tramos rectos.

### Verificación tensional y factores de seguridad

La tensión normal en una fibra extrema se calcula con la fórmula de Navier:

$$
\sigma(x) = \frac{N(x)}{A} \pm \frac{M(x)\,c}{I},
$$

siendo $c$ la distancia desde el eje neutro hasta la fibra considerada. Se
adopta el máximo valor absoluto en todas las estaciones de cálculo y se compara
con el límite elástico $S_y$:

$$
FS = \frac{S_y}{\left|\sigma\right|_{max}}.
$$

El estado de la sección se informa según el factor de seguridad admisible
$FS_{adm}$:

- **OK:** $FS \geq FS_{adm}$.
- **ALERTA:** $1.0 \leq FS < FS_{adm}$.
- **FLUENCIA:** $FS < 1.0$.

Estos estados describen la resistencia normal elástica de la sección y no
incluyen verificaciones de inestabilidad o de interacción entre modos de falla.

### Referencias bibliográficas

1. S. P. Timoshenko y J. M. Gere, *Theory of Elastic Stability* y textos de
   resistencia de materiales y teoría de estructuras.
2. W. McGuire, R. H. Gallagher y R. D. Saunders, *Matrix Structural Analysis*.
3. R. H. Gallagher y A. J. Ziemian, *Matrix Structural Analysis*.
"""
    )
