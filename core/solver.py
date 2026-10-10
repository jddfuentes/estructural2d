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
    reactions: dict[int, tuple[float, float, float]]  # nodo -> (Rx [N], Ry [N], Mz [N·mm])
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


def local_stiffness(E: float, A: float, I: float, L: float) -> FloatArray:  # noqa: E741
    """Matriz de rigidez local 6x6 del elemento de pórtico."""
    ea, ei = E * A / L, E * I
    k1, k2, k3, k4 = 12 * ei / L**3, 6 * ei / L**2, 4 * ei / L, 2 * ei / L
    return np.array(
        [
            [ea, 0, 0, -ea, 0, 0],
            [0, k1, k2, 0, -k1, k2],
            [0, k2, k3, 0, -k2, k4],
            [-ea, 0, 0, ea, 0, 0],
            [0, -k1, -k2, 0, k1, -k2],
            [0, k2, k4, 0, -k2, k3],
        ],
        dtype=np.float64,
    )


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
        k = local_stiffness(mem.material.E, mem.section.A, mem.section.I, L)
        T = rotation_matrix(c, s)
        dofs = np.r_[3 * mem.i : 3 * mem.i + 3, 3 * mem.j : 3 * mem.j + 3]
        K[np.ix_(dofs, dofs)] += T.T @ k @ T
        qx1, qx2, qy1, qy2 = member_q[m]
        f_eq = equivalent_nodal_loads((qx1, qx2), (qy1, qy2), L)
        F[dofs] += T.T @ f_eq
        elem.append((k, T, f_eq, dofs))

    for p in model.nodal_loads:
        F[3 * p.node : 3 * p.node + 3] += (p.Fx, p.Fy, p.Mz)

    # Condiciones de borde
    restrained = np.zeros(ndof, dtype=bool)
    for sup in model.supports:
        restrained[3 * sup.node : 3 * sup.node + 3] = sup.type.restrained_dofs
    free = ~restrained

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
    R = K @ U - F

    reactions = {
        sup.node: (float(R[3 * sup.node]), float(R[3 * sup.node + 1]), float(R[3 * sup.node + 2]))
        for sup in model.supports
    }

    members = [
        _member_results(model, m, k, T, f_eq, U[dofs], member_q[m], n_stations)
        for m, (k, T, f_eq, dofs) in enumerate(elem)
    ]
    return Results(displacements=U.reshape(-1, 3), reactions=reactions, members=members)


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
    f = k @ ul - f_eq  # fuerzas de extremo sobre la barra (locales)
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
