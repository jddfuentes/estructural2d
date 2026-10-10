"""Solver matricial de pórticos planos 2D (método directo de rigidez).

Función pública principal: `solve(model) -> Results` (pura, sin estado, sin UI).

Formulación
-----------
- Elemento de pórtico Euler-Bernoulli, 3 GDL por nodo (ux, uy, rz).
- Cargas distribuidas lineales -> cargas nodales equivalentes consistentes
  (integración de Gauss de las funciones de forma de Hermite).
- Solicitaciones internas N, V, M por equilibrio exacto de cada barra
  (polinomios), por lo que son exactas aún con una sola barra por tramo.
- Elástica dentro de cada barra por doble integración exacta de M/EI con
  las condiciones de borde nodales (exacta para Euler-Bernoulli).
- Rótulas internas (`Member.release_start/release_end`): condensación estática local del
  giro liberado. Matriz en forma cerrada (3EI/L³, 3EI/L², 3EI/L; biela: sólo EA/L) y cargas
  equivalentes condensadas f_c = f_r − k_rc·k_cc⁻¹·f_c (exactas también para carga trapezoidal).
  Un nudo donde TODAS las barras concurrentes están articuladas no tiene rigidez a giro: su GDL
  rz se excluye del sistema y se informa 0 (cada barra gira por su cuenta).
- Apoyos elásticos (`Model.springs`): kx, ky, krz se suman a la diagonal de K global en los GDL
  (ux, uy, rz) del nodo. Un GDL con resorte y sin restricción rígida es libre. Reacción del
  resorte sobre la estructura: R = −k·u, sumada a la del vínculo rígido del mismo nudo (si hay).
  Un resorte rotacional en un nudo totalmente articulado le devuelve la rigidez a giro.

Convención de signos de solicitaciones (ejes locales de cada barra, x de i a j):
- N > 0 tracción.
- M > 0 tracciona la fibra inferior (y local negativa) -> "sagging".
- V = dM/dx.
Unidades: mm, N, MPa, N·mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial as P
from numpy.typing import NDArray

from core.model import LoadDirection, Model

FloatArray = NDArray[np.float64]

_COND_LIMIT = 1e12  # número de condición (matriz escalada) para declarar mecanismo
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


class StructuralError(Exception):
    """Error de resolución (estructura hipostática / mecanismo, etc.)."""


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MemberResult:
    """Resultados muestreados a lo largo de una barra."""

    member: int
    length: float
    x: FloatArray  # estaciones locales [mm], 0..L
    points: FloatArray  # (n, 2) coordenadas globales sin deformar [mm]
    N: FloatArray  # esfuerzo normal [N]
    V: FloatArray  # esfuerzo de corte [N]
    M: FloatArray  # momento flector [N·mm]
    u_local: FloatArray  # desplazamiento axial local [mm]
    v_local: FloatArray  # desplazamiento transversal local [mm]
    displacement: FloatArray  # (n, 2) desplazamiento global (ux, uy) [mm]
    sigma_top: FloatArray  # tensión normal fibra superior (y local +) [MPa]
    sigma_bot: FloatArray  # tensión normal fibra inferior [MPa]
    end_forces: FloatArray  # (6,) fuerzas de extremo locales sobre la barra

    @property
    def sigma_abs(self) -> FloatArray:
        """|σ| máxima de la sección en cada estación (normal + flexión) [MPa]."""
        out: FloatArray = np.maximum(np.abs(self.sigma_top), np.abs(self.sigma_bot))
        return out

    @property
    def deflection(self) -> FloatArray:
        """Módulo del desplazamiento total en cada estación [mm]."""
        return np.hypot(self.displacement[:, 0], self.displacement[:, 1])


@dataclass(frozen=True, slots=True)
class Extreme:
    """Valor extremo con su ubicación."""

    value: float  # valor con signo
    member: int
    x: float  # posición local en la barra [mm]
    point: tuple[float, float]  # posición global [mm]


@dataclass(frozen=True, slots=True)
class Results:
    displacements: FloatArray  # (n_nodos, 3): ux [mm], uy [mm], rz [rad]
    # nodo con apoyo rígido y/o resorte -> (Rx [N], Ry [N], Mz [N·mm]); incluye la fuerza −k·u
    reactions: dict[int, tuple[float, float, float]]
    members: list[MemberResult] = field(default_factory=list)

    def extreme(self, quantity: str) -> Extreme:
        """Máximo en valor absoluto de: 'N', 'V', 'M', 'sigma', 'deflection', 'uy'."""
        best: Extreme | None = None
        for mr in self.members:
            arr = _quantity(mr, quantity)
            k = int(np.argmax(np.abs(arr)))
            if best is None or abs(arr[k]) > abs(best.value):
                best = Extreme(
                    value=float(arr[k]), member=mr.member, x=float(mr.x[k]),
                    point=(float(mr.points[k, 0]), float(mr.points[k, 1])),
                )
        if best is None:
            raise ValueError("Resultados sin barras.")
        return best


def _quantity(mr: MemberResult, q: str) -> FloatArray:
    match q:
        case "N":
            return mr.N
        case "V":
            return mr.V
        case "M":
            return mr.M
        case "sigma":
            return mr.sigma_abs
        case "deflection":
            return mr.deflection
        case "uy":
            return mr.displacement[:, 1]
        case _:
            raise ValueError(f"Magnitud desconocida: {q!r}")


# --------------------------------------------------------------------------- #
# Elemento
# --------------------------------------------------------------------------- #


_RZ_I, _RZ_J = 2, 5  # índices locales de los giros en (u_i, v_i, θ_i, u_j, v_j, θ_j)


def released_dofs(release_start: bool, release_end: bool) -> tuple[int, ...]:
    """Índices locales de los giros liberados (rótulas) de una barra."""
    return tuple(d for d, flag in ((_RZ_I, release_start), (_RZ_J, release_end)) if flag)


def local_stiffness(
    E: float,
    A: float,
    I: float,  # noqa: E741
    L: float,
    release_start: bool = False,
    release_end: bool = False,
) -> FloatArray:
    """Matriz de rigidez local 6x6 del elemento de pórtico.

    Con rótulas devuelve la matriz condensada estáticamente en forma cerrada (Gere & Weaver,
    "Analysis of Framed Structures", cap. 4): fila y columna del giro liberado nulas.
    - Rótula en i: términos 3EI/L³ (corte), 3EI/L² y 3EI/L en θ_j.
    - Rótula en j: simétrica, 3EI/L en θ_i.
    - Rótula en ambos (biela): flexión y corte nulos, sólo EA/L.
    """
    ea, ei = E * A / L, E * I
    if release_start and release_end:
        k1 = k2 = k3 = k4 = k5 = k6 = 0.0
    elif release_start or release_end:
        a, b, c = 3 * ei / L**3, 3 * ei / L**2, 3 * ei / L
        k1 = a
        k2, k5 = (0.0, b) if release_start else (b, 0.0)  # acoples v-θ en i / en j
        k3, k6 = (0.0, c) if release_start else (c, 0.0)  # θ_i-θ_i / θ_j-θ_j
        k4 = 0.0  # acople θ_i-θ_j
    else:
        k1, k2, k3, k4 = 12 * ei / L**3, 6 * ei / L**2, 4 * ei / L, 2 * ei / L
        k5, k6 = k2, k3
    return np.array(
        [
            [ea, 0, 0, -ea, 0, 0],
            [0, k1, k2, 0, -k1, k5],
            [0, k2, k3, 0, -k2, k4],
            [-ea, 0, 0, ea, 0, 0],
            [0, -k1, -k2, 0, k1, -k5],
            [0, k5, k4, 0, -k5, k6],
        ],
        dtype=np.float64,
    )


def static_condensation(
    k: FloatArray, f: FloatArray, released: tuple[int, ...]
) -> tuple[FloatArray, FloatArray]:
    """Condensación estática de los GDL locales `released` (fuerza nula en ellos).

    k_c = k_rr − k_rc·k_cc⁻¹·k_cr ;  f_c = f_r − k_rc·k_cc⁻¹·f_c, devueltas en 6x6 / 6 con
    ceros exactos en filas, columnas y componentes condensadas.
    """
    if not released:
        return k, f
    rel = list(released)
    keep = [d for d in range(k.shape[0]) if d not in released]
    X = np.linalg.solve(k[np.ix_(rel, rel)], k[np.ix_(rel, keep)])  # k_cc⁻¹·k_cr
    kc = np.zeros_like(k)
    fc = np.zeros_like(f)
    kc[np.ix_(keep, keep)] = k[np.ix_(keep, keep)] - k[np.ix_(keep, rel)] @ X
    fc[keep] = f[keep] - X.T @ f[rel]  # k simétrica: k_rc·k_cc⁻¹ = (k_cc⁻¹·k_cr)ᵀ
    return kc, fc


def rotation_matrix(c: float, s: float) -> FloatArray:
    """T tal que u_local = T @ u_global (6x6)."""
    r = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.zeros((6, 6))
    T[:3, :3] = r
    T[3:, 3:] = r
    return T


def _local_load_components(
    q1: float, q2: float, direction: LoadDirection, c: float, s: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Proyecta una carga distribuida a ejes locales -> ((qx1, qx2), (qy1, qy2))."""
    match direction:
        case LoadDirection.LOCAL_Y:
            fx, fy = 0.0, 1.0
        case LoadDirection.GLOBAL_Y:  # vector global (0, q) -> local (q·s, q·c)
            fx, fy = s, c
        case LoadDirection.GLOBAL_X:  # vector global (q, 0) -> local (q·c, -q·s)
            fx, fy = c, -s
    return (q1 * fx, q2 * fx), (q1 * fy, q2 * fy)


def equivalent_nodal_loads(qx: tuple[float, float], qy: tuple[float, float], L: float) -> FloatArray:
    """Vector de cargas nodales equivalentes (locales) para carga lineal qx, qy."""
    xi = 0.5 * (_GAUSS_X + 1.0)
    w = 0.5 * _GAUSS_W * L
    qxv = qx[0] + (qx[1] - qx[0]) * xi
    qyv = qy[0] + (qy[1] - qy[0]) * xi
    N = np.array(
        [
            1 - xi,
            1 - 3 * xi**2 + 2 * xi**3,
            L * (xi - 2 * xi**2 + xi**3),
            xi,
            3 * xi**2 - 2 * xi**3,
            L * (-(xi**2) + xi**3),
        ]
    )
    q = np.array([qxv, qyv, qyv, qxv, qyv, qyv])
    return np.asarray((N * q) @ w, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


def solve(model: Model, n_stations: int = 41) -> Results:
    """Resuelve el pórtico plano. Función pura: no modifica `model`.

    `n_stations`: estaciones uniformes por barra para el postproceso (≥ 2: ambos extremos).
    A ellas se suman siempre las abscisas exactas de M extremo (V = 0) y de flecha extrema.

    Raises:
        ValueError: modelo inconsistente o `n_stations` < 2.
        StructuralError: estructura inestable (mecanismo / hipostática).
    """
    if n_stations < 2:
        raise ValueError("La cantidad de estaciones debe ser al menos 2.")
    model.validate()
    ndof = model.n_dof
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # Cargas distribuidas agrupadas por barra en ejes locales (qx1, qx2, qy1, qy2)
    member_q = np.zeros((len(model.members), 4))
    for dl in model.distributed_loads:
        c, s = model.member_cos_sin(dl.member)
        (qx1, qx2), (qy1, qy2) = _local_load_components(dl.q_start, dl.q_end, dl.direction, c, s)
        member_q[dl.member] += (qx1, qx2, qy1, qy2)

    # Ensamblaje
    elem: list[tuple[FloatArray, FloatArray, FloatArray, NDArray[np.intp]]] = []
    for m, mem in enumerate(model.members):
        L = model.member_length(m)
        c, s = model.member_cos_sin(m)
        E, A, I = mem.material.E, mem.section.A, mem.section.I  # noqa: E741
        k = local_stiffness(E, A, I, L, mem.release_start, mem.release_end)
        T = rotation_matrix(c, s)
        dofs = np.r_[3 * mem.i : 3 * mem.i + 3, 3 * mem.j : 3 * mem.j + 3]
        K[np.ix_(dofs, dofs)] += T.T @ k @ T
        qx1, qx2, qy1, qy2 = member_q[m]
        f_eq = equivalent_nodal_loads((qx1, qx2), (qy1, qy2), L)
        released = released_dofs(mem.release_start, mem.release_end)
        if released:  # cargas de empotramiento de la barra con rótula (p.ej. 3qL/8, 5qL/8, qL²/8)
            f_eq = static_condensation(local_stiffness(E, A, I, L), f_eq, released)[1]
        F[dofs] += T.T @ f_eq
        elem.append((k, T, f_eq, dofs))

    for p in model.nodal_loads:
        F[3 * p.node : 3 * p.node + 3] += (p.Fx, p.Fy, p.Mz)

    # Apoyos elásticos: rigidez de resorte en la diagonal (GDL desacoplados, ejes globales)
    k_spring = np.zeros(ndof)
    for sp in model.springs:
        k_spring[3 * sp.node : 3 * sp.node + 3] += sp.stiffness
    K[np.diag_indices(ndof)] += k_spring

    # Condiciones de borde
    restrained = np.zeros(ndof, dtype=bool)
    for sup in model.supports:
        restrained[3 * sup.node : 3 * sup.node + 3] = sup.type.restrained_dofs
    free = ~restrained

    # Nudos totalmente articulados: rz sin rigidez -> fuera del sistema (giro nodal informado = 0).
    # Con resorte rotacional (krz > 0) el giro del nudo sí tiene rigidez y queda en el sistema.
    hinged = {n for n in _fully_hinged_nodes(model) if k_spring[3 * n + 2] == 0.0}
    for p in model.nodal_loads:
        if p.node in hinged and p.Mz != 0.0:
            raise StructuralError(
                f"Nodo {p.node}: momento aplicado en un nudo donde todas las barras tienen rótula; "
                "no hay rigidez a giro que lo resista. Quitar una rótula, aplicar el momento en "
                "un nudo con alguna barra continua o agregar un resorte rotacional (krz) en el nudo."
            )
    for n in hinged:
        free[3 * n + 2] = False

    U = np.zeros(ndof)
    if free.any():  # (puede no haber GDL libres: p.ej. barra biempotrada sin nodos intermedios)
        Kff = K[np.ix_(free, free)]
        diag = np.diag(Kff)
        if np.any(diag <= 0):
            raise StructuralError("Hay GDL sin rigidez: revisar nodos sueltos o barras desconectadas.")
        d = 1.0 / np.sqrt(diag)
        Ks = Kff * np.outer(d, d)  # escalado diagonal -> condición independiente de unidades
        if np.linalg.cond(Ks) > _COND_LIMIT:
            raise StructuralError(
                "Estructura inestable (mecanismo): revisar apoyos. "
                "Ej.: viga con un solo apoyo, o sólo apoyos móviles sin restricción horizontal."
            )
        U[free] = d * np.linalg.solve(Ks, d * F[free])
    # Reacciones sobre la estructura. Vínculo rígido: K·U − F (K incluye resortes, pero u = 0 ahí).
    # GDL libre con resorte: R = −k·u (opuesta al desplazamiento), asignada exacta; equivale a
    # K_barras·U − F salvo el residuo de la solución, así que ΣR + ΣF = 0 se cumple al redondeo.
    R = K @ U - F - k_spring * U
    spring_free = free & (k_spring > 0.0)
    R[spring_free] = -k_spring[spring_free] * U[spring_free]

    support_nodes = [sup.node for sup in model.supports]
    spring_nodes = [sp.node for sp in model.springs if sp.node not in support_nodes]
    reactions = {
        n: (float(R[3 * n]), float(R[3 * n + 1]), float(R[3 * n + 2]))
        for n in support_nodes + spring_nodes
    }

    members = [
        _member_results(model, m, k, T, f_eq, U[dofs], member_q[m], n_stations)
        for m, (k, T, f_eq, dofs) in enumerate(elem)
    ]
    return Results(displacements=U.reshape(-1, 3), reactions=reactions, members=members)


def _fully_hinged_nodes(model: Model) -> set[int]:
    """Nodos con al menos una barra donde TODAS las barras concurrentes tienen rótula en él."""
    touched: set[int] = set()
    rigid: set[int] = set()
    for mem in model.members:
        touched.update((mem.i, mem.j))
        if not mem.release_start:
            rigid.add(mem.i)
        if not mem.release_end:
            rigid.add(mem.j)
    return touched - rigid


def _member_results(
    model: Model,
    m: int,
    k: FloatArray,
    T: FloatArray,
    f_eq: FloatArray,
    u_global: FloatArray,
    q: FloatArray,
    n_stations: int,
) -> MemberResult:
    mem = model.members[m]
    L = model.member_length(m)
    c, s = model.member_cos_sin(m)
    E, sec = mem.material.E, mem.section
    ul = T @ u_global
    # Fuerzas de extremo sobre la barra (locales). Con rótula, k y f_eq están condensadas:
    # la componente de momento liberada es 0.0 exacto y el giro nodal no interviene.
    f = k @ ul - f_eq
    qx1, qx2, qy1, qy2 = q

    # Polinomios exactos en x (local)
    qx = P([qx1, (qx2 - qx1) / L])
    qy = P([qy1, (qy2 - qy1) / L])
    Npoly = -f[0] - qx.integ()
    Vpoly = f[1] + qy.integ()
    Mpoly = P([-f[2]]) + Vpoly.integ()  # dM/dx = V, M(0) = -M_i

    # Elástica: u' = N/EA ; v'' = M/EI ; con v(0)=v1, v(L)=v2
    upoly = ul[0] + (Npoly / (E * sec.A)).integ()
    m2 = (Mpoly / (E * sec.I)).integ(2)  # particular con v(0)=v'(0)=0
    c1 = (ul[4] - ul[1] - m2(L)) / L
    vpoly = m2 + P([ul[1], c1])

    # Estaciones: uniformes + extremos exactos de M (V = 0) y de la flecha (v' = 0)
    xs = np.linspace(0.0, L, n_stations)
    x = np.unique(np.concatenate([xs, _roots_in(Vpoly, L), _roots_in(vpoly.deriv(), L)]))

    Nx, Vx, Mx = Npoly(x), Vpoly(x), Mpoly(x)
    # Rótulas: M = 0 exacto en el extremo liberado (x = 0 / x = L son siempre estaciones).
    # M(0) = −f[2] ya es 0.0; M(L) se fija para eliminar el redondeo de evaluar el polinomio.
    if mem.release_start:
        Mx[0] = 0.0
    if mem.release_end:
        Mx[-1] = 0.0
    u, v = upoly(x), vpoly(x)
    (x1, y1) = model.nodes[mem.i]
    points = np.column_stack([x1 + c * x, y1 + s * x])
    disp = np.column_stack([c * u - s * v, s * u + c * v])

    sN = Nx / sec.A
    return MemberResult(
        member=m, length=L, x=x, points=points,
        N=Nx, V=Vx, M=Mx, u_local=u, v_local=v, displacement=disp,
        sigma_top=sN - Mx * sec.c_top / sec.I,
        sigma_bot=sN + Mx * sec.c_bot / sec.I,
        end_forces=f,
    )


def _roots_in(p: P, L: float) -> FloatArray:
    """Raíces reales de p en el intervalo abierto (0, L)."""
    if p.degree() < 1:
        return np.empty(0)
    r = p.roots()
    r = r[np.abs(r.imag) <= 1e-9 * max(L, 1.0)].real
    return np.asarray(r[(r > 0.0) & (r < L)], dtype=np.float64)
