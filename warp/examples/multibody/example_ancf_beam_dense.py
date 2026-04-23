"""Dense-first Warp ANCF beam example."""

import argparse
import os
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA solve path
    cp = None

import warp as wp

DEFAULT_OUTPUT_DIR = "."


# -----------------------------------------------------------------------------
# Quadrature and ANCF shape functions
# -----------------------------------------------------------------------------


GL_TABLE = {
    2: {
        "x": np.array([-0.5773502691896257, 0.5773502691896257]),
        "w": np.array([1.0000000000000000, 1.0000000000000000]),
    },
    4: {
        "x": np.array([-0.3399810435848563, 0.3399810435848563, -0.8611363115940526, 0.8611363115940526]),
        "w": np.array([0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0.3478548451374538]),
    },
}


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the tabulated Gauss-Legendre rule."""
    data = GL_TABLE[n]
    return data["x"], data["w"]


def eval_basis(u: float, v: float, w: float) -> np.ndarray:
    """Evaluate the scalar ANCF polynomial basis."""
    return np.array([1.0, u, v, w, u * v, u * w, u**2, u**3], dtype=np.float64)


def eval_basis_derivatives(u: float, v: float, w: float) -> np.ndarray:
    """Evaluate the basis derivatives with respect to ``u``, ``v``, and ``w``."""
    ddu = np.array([0.0, 1.0, 0.0, 0.0, v, w, 2.0 * u, 3.0 * u**2], dtype=np.float64)
    ddv = np.array([0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0, 0.0], dtype=np.float64)
    ddw = np.array([0.0, 0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0], dtype=np.float64)
    return np.array([ddu, ddv, ddw], dtype=np.float64)


def build_B_matrix(l_elem: float) -> np.ndarray:
    """Build the interpolation matrix for one beam element."""
    B = np.zeros((8, 8), dtype=np.float64)

    u1 = -l_elem / 2.0
    B[0, :] = eval_basis(u1, 0.0, 0.0)
    B[1, :], B[2, :], B[3, :] = eval_basis_derivatives(u1, 0.0, 0.0)

    u2 = l_elem / 2.0
    B[4, :] = eval_basis(u2, 0.0, 0.0)
    B[5, :], B[6, :], B[7, :] = eval_basis_derivatives(u2, 0.0, 0.0)
    return B


def shape_functions(u: float, v: float, w: float, B_inv: np.ndarray) -> np.ndarray:
    """Return the 8 nodal shape functions at ``(u, v, w)``."""
    return B_inv @ eval_basis(u, v, w)


# -----------------------------------------------------------------------------
# Beam setup and host precompute data
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussData:
    weights: np.ndarray
    S: np.ndarray
    H: np.ndarray


@dataclass(frozen=True)
class SimContext:
    n_beam: int
    N_coef: int
    n_dofs: int
    n_constraints: int
    L_elem: float
    W: float
    H: float
    rho: float
    E: float
    nu: float
    lam: float
    mu: float
    elem_conn: np.ndarray
    B_inv: np.ndarray
    gauss: GaussData


def _precompute_element_quadrature(l_elem: float, width: float, height: float, B_inv: np.ndarray) -> GaussData:
    j_det = (l_elem * width * height) / 8.0

    gp_u, w_u = gauss_legendre(4)
    gp_v, w_v = gauss_legendre(2)
    gp_w, w_w = gauss_legendre(2)

    weights_list: list[float] = []
    S_list: list[np.ndarray] = []
    H_list: list[np.ndarray] = []

    for i_u, xi in enumerate(gp_u):
        u = (l_elem / 2.0) * xi
        for i_v, eta in enumerate(gp_v):
            v = (width / 2.0) * eta
            for i_w, zeta in enumerate(gp_w):
                w = (height / 2.0) * zeta
                weights_list.append(w_u[i_u] * w_v[i_v] * w_w[i_w] * j_det)
                basis = eval_basis(u, v, w)
                basis_derivs = eval_basis_derivatives(u, v, w)
                S_list.append(B_inv @ basis)
                H_list.append(B_inv @ basis_derivs.T)

    return GaussData(
        weights=np.array(weights_list, dtype=np.float64),
        S=np.array(S_list, dtype=np.float64),
        H=np.array(H_list, dtype=np.float64),
    )


def build_mesh(
    n_elements: int,
    l_total: float,
    width: float = 0.003,
    height: float = 0.003,
    rho: float = 7700.0,
    E: float = 2.0e11,
    nu: float = 0.3,
) -> tuple[SimContext, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a straight cantilever mesh and split initial state."""
    l_elem = l_total / n_elements
    n_beam = n_elements
    n_coef = 8 + 4 * (n_beam - 1)
    n_dofs = 3 * n_coef

    elem_conn = 4 * np.arange(n_beam, dtype=np.int64)[:, None] + np.arange(8, dtype=np.int64)[None, :]
    B_inv = np.linalg.inv(build_B_matrix(l_elem).T)
    gauss = _precompute_element_quadrature(l_elem, width, height, B_inv)

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    ctx = SimContext(
        n_beam=n_beam,
        N_coef=n_coef,
        n_dofs=n_dofs,
        n_constraints=12,
        L_elem=l_elem,
        W=width,
        H=height,
        rho=rho,
        E=E,
        nu=nu,
        lam=lam,
        mu=mu,
        elem_conn=elem_conn,
        B_inv=B_inv,
        gauss=gauss,
    )

    k = np.arange(n_coef, dtype=np.int64)
    g = k // 4
    r = k % 4

    x0 = np.where(r == 0, g * l_elem, np.where(r == 1, 1.0, 0.0)).astype(np.float64)
    y0 = np.where(r == 2, 1.0, 0.0).astype(np.float64)
    z0 = np.where(r == 3, 1.0, 0.0).astype(np.float64)
    vx0 = np.zeros(n_coef, dtype=np.float64)
    vy0 = np.zeros(n_coef, dtype=np.float64)
    vz0 = np.zeros(n_coef, dtype=np.float64)

    print(f"Element Initialized: Length={l_elem:.4f}m, J_det={(l_elem * width * height) / 8.0:.2e}")
    return ctx, x0, y0, z0, vx0, vy0, vz0


def F_tip(t: float) -> np.ndarray:
    """Return the time-dependent tip load."""
    if 0.0 <= t <= 0.05:
        force_z = -1.0 + np.cos(20.0 * np.pi * t)
        return np.array([0.0, 0.0, force_z], dtype=np.float64)
    return np.array([0.0, 0.0, 0.0], dtype=np.float64)


def precompute_mass_matrix(ctx: SimContext) -> np.ndarray:
    """Build the global consistent mass matrix."""
    S = ctx.gauss.S
    weights = ctx.gauss.weights
    m_global = np.zeros((ctx.N_coef, ctx.N_coef), dtype=np.float64)

    for elem in range(ctx.n_beam):
        m_elem = np.zeros((8, 8), dtype=np.float64)
        for gp_idx in range(16):
            s = S[gp_idx, :]
            weight = weights[gp_idx]
            m_elem += ctx.rho * np.outer(s, s) * weight

        m_elem = 0.5 * (m_elem + m_elem.T)
        conn = ctx.elem_conn[elem]
        m_global[np.ix_(conn, conn)] += m_elem

    return np.kron(m_global, np.eye(3, dtype=np.float64))


def precompute_gravity_force(ctx: SimContext, g_vec: np.ndarray) -> np.ndarray:
    """Build the global gravity force vector."""
    S = ctx.gauss.S
    weights = ctx.gauss.weights
    G_global = np.zeros((ctx.N_coef, 3), dtype=np.float64)

    for elem in range(ctx.n_beam):
        v_i = np.zeros(8, dtype=np.float64)
        for gp_idx in range(16):
            s = S[gp_idx, :]
            weight = weights[gp_idx]
            v_i += s * weight

        G_elem = ctx.rho * np.outer(v_i, g_vec)
        conn = ctx.elem_conn[elem]
        G_global[conn, :] += G_elem

    return G_global.reshape(-1)


# -----------------------------------------------------------------------------
# Boundary conditions and loading
# -----------------------------------------------------------------------------


CLAMP_TARGET = np.array(
    [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Runtime state containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DensePrecompute:
    ctx: SimContext
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    vx0: np.ndarray
    vy0: np.ndarray
    vz0: np.ndarray
    j_mass: np.ndarray
    r_gravity: np.ndarray
    tip_shape8: np.ndarray


@wp.struct
class DenseWarpBuffers:
    n_coef: int
    n_dofs: int
    elem_conn: wp.array2d[wp.int32]
    weights: wp.array[wp.float64]
    H: wp.array3d[wp.float64]
    j_mass: wp.array2d[wp.float64]
    r_gravity: wp.array[wp.float64]
    tip_shape8: wp.array[wp.float64]
    x: wp.array[wp.float64]
    y: wp.array[wp.float64]
    z: wp.array[wp.float64]
    vx: wp.array[wp.float64]
    vy: wp.array[wp.float64]
    vz: wp.array[wp.float64]
    ax: wp.array[wp.float64]
    ay: wp.array[wp.float64]
    az: wp.array[wp.float64]
    x_prev: wp.array[wp.float64]
    y_prev: wp.array[wp.float64]
    z_prev: wp.array[wp.float64]
    vx_prev: wp.array[wp.float64]
    vy_prev: wp.array[wp.float64]
    vz_prev: wp.array[wp.float64]
    J: wp.array2d[wp.float64]
    R: wp.array[wp.float64]
    delta_a: wp.array[wp.float64]


@dataclass
class DenseWarpState:
    device: wp.Device
    buffers: DenseWarpBuffers

    @property
    def n_coef(self) -> int:
        return self.buffers.n_coef

    @property
    def n_dofs(self) -> int:
        return self.buffers.n_dofs


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------


@wp.kernel
def save_prev_state_kernel(buffers: DenseWarpBuffers):
    i = wp.tid()
    buffers.x_prev[i] = buffers.x[i]
    buffers.y_prev[i] = buffers.y[i]
    buffers.z_prev[i] = buffers.z[i]
    buffers.vx_prev[i] = buffers.vx[i]
    buffers.vy_prev[i] = buffers.vy[i]
    buffers.vz_prev[i] = buffers.vz[i]


@wp.kernel
def compute_trial_position_kernel(buffers: DenseWarpBuffers, dt: wp.float64):
    i = wp.tid()
    dt_sq = dt * dt
    buffers.x[i] = buffers.x_prev[i] + dt * buffers.vx_prev[i] + dt_sq * buffers.ax[i]
    buffers.y[i] = buffers.y_prev[i] + dt * buffers.vy_prev[i] + dt_sq * buffers.ay[i]
    buffers.z[i] = buffers.z_prev[i] + dt * buffers.vz_prev[i] + dt_sq * buffers.az[i]


@wp.kernel
def initialize_residual_kernel(buffers: DenseWarpBuffers):
    row = wp.tid()
    accum = wp.float64(0.0)

    for col in range(buffers.n_dofs):
        coef = col // 3
        comp = col - 3 * coef
        a_val = wp.float64(0.0)
        if comp == 0:
            a_val = buffers.ax[coef]
        elif comp == 1:
            a_val = buffers.ay[coef]
        else:
            a_val = buffers.az[coef]

        accum += buffers.j_mass[row, col] * a_val

    buffers.R[row] = accum - buffers.r_gravity[row]


@wp.kernel
def apply_tip_load_kernel(
    buffers: DenseWarpBuffers,
    last_elem: int,
    fx: wp.float64,
    fy: wp.float64,
    fz: wp.float64,
):
    i = wp.tid()
    coef = buffers.elem_conn[last_elem, i]
    weight = buffers.tip_shape8[i]

    buffers.R[3 * coef + 0] -= weight * fx
    buffers.R[3 * coef + 1] -= weight * fy
    buffers.R[3 * coef + 2] -= weight * fz


@wp.kernel
def clamp_columns_kernel(buffers: DenseWarpBuffers):
    col = wp.tid()
    for row in range(buffers.n_dofs):
        buffers.J[row, col] = wp.float64(0.0)


# Keep clamped DOFs in the dense system for simple global indexing. Zero their
# residual, clear their matrix columns, and set identity rows so the solve gives
# zero updates, matching a reduced free-DOF solve without reduced indexing.
@wp.kernel
def clamp_residual_kernel(buffers: DenseWarpBuffers):
    row = wp.tid()
    buffers.R[row] = wp.float64(0.0)


@wp.kernel
def clamp_matrix_rows_kernel(buffers: DenseWarpBuffers):
    row = wp.tid()
    for col in range(buffers.n_dofs):
        buffers.J[row, col] = wp.float64(0.0)

    buffers.J[row, row] = wp.float64(1.0)


@wp.kernel
def update_acceleration_from_delta_kernel(buffers: DenseWarpBuffers):
    coef = wp.tid()
    buffers.ax[coef] += buffers.delta_a[3 * coef + 0]
    buffers.ay[coef] += buffers.delta_a[3 * coef + 1]
    buffers.az[coef] += buffers.delta_a[3 * coef + 2]


@wp.kernel
def accept_step_kernel(buffers: DenseWarpBuffers, dt: wp.float64):
    coef = wp.tid()
    buffers.vx[coef] = buffers.vx_prev[coef] + dt * buffers.ax[coef]
    buffers.vy[coef] = buffers.vy_prev[coef] + dt * buffers.ay[coef]
    buffers.vz[coef] = buffers.vz_prev[coef] + dt * buffers.az[coef]


@wp.kernel
def record_tip_kernel(
    buffers: DenseWarpBuffers,
    tip_history: wp.array2d(dtype=wp.float64),
    out_index: int,
    last_elem: int,
):
    tip_x = wp.float64(0.0)
    tip_y = wp.float64(0.0)
    tip_z = wp.float64(0.0)

    for i in range(8):
        coef = buffers.elem_conn[last_elem, i]
        shape = buffers.tip_shape8[i]
        tip_x += shape * buffers.x[coef]
        tip_y += shape * buffers.y[coef]
        tip_z += shape * buffers.z[coef]

    tip_history[out_index, 0] = tip_x
    tip_history[out_index, 1] = tip_y
    tip_history[out_index, 2] = tip_z


@wp.kernel
def assemble_internal_force_kernel(
    buffers: DenseWarpBuffers,
    lam: wp.float64,
    mu: wp.float64,
):
    # Keep the constitutive math expanded into scalar component operations here.
    # This was chosen to keep Warp codegen predictable and avoid extra temporaries.
    e = wp.tid()

    conn0 = buffers.elem_conn[e, 0]
    conn1 = buffers.elem_conn[e, 1]
    conn2 = buffers.elem_conn[e, 2]
    conn3 = buffers.elem_conn[e, 3]
    conn4 = buffers.elem_conn[e, 4]
    conn5 = buffers.elem_conn[e, 5]
    conn6 = buffers.elem_conn[e, 6]
    conn7 = buffers.elem_conn[e, 7]

    x0 = buffers.x[conn0]
    x1 = buffers.x[conn1]
    x2 = buffers.x[conn2]
    x3 = buffers.x[conn3]
    x4 = buffers.x[conn4]
    x5 = buffers.x[conn5]
    x6 = buffers.x[conn6]
    x7 = buffers.x[conn7]

    y0 = buffers.y[conn0]
    y1 = buffers.y[conn1]
    y2 = buffers.y[conn2]
    y3 = buffers.y[conn3]
    y4 = buffers.y[conn4]
    y5 = buffers.y[conn5]
    y6 = buffers.y[conn6]
    y7 = buffers.y[conn7]

    z0 = buffers.z[conn0]
    z1 = buffers.z[conn1]
    z2 = buffers.z[conn2]
    z3 = buffers.z[conn3]
    z4 = buffers.z[conn4]
    z5 = buffers.z[conn5]
    z6 = buffers.z[conn6]
    z7 = buffers.z[conn7]

    f0x = wp.float64(0.0)
    f0y = wp.float64(0.0)
    f0z = wp.float64(0.0)
    f1x = wp.float64(0.0)
    f1y = wp.float64(0.0)
    f1z = wp.float64(0.0)
    f2x = wp.float64(0.0)
    f2y = wp.float64(0.0)
    f2z = wp.float64(0.0)
    f3x = wp.float64(0.0)
    f3y = wp.float64(0.0)
    f3z = wp.float64(0.0)
    f4x = wp.float64(0.0)
    f4y = wp.float64(0.0)
    f4z = wp.float64(0.0)
    f5x = wp.float64(0.0)
    f5y = wp.float64(0.0)
    f5z = wp.float64(0.0)
    f6x = wp.float64(0.0)
    f6y = wp.float64(0.0)
    f6z = wp.float64(0.0)
    f7x = wp.float64(0.0)
    f7y = wp.float64(0.0)
    f7z = wp.float64(0.0)

    for gp in range(16):
        h00 = buffers.H[gp, 0, 0]
        h01 = buffers.H[gp, 0, 1]
        h02 = buffers.H[gp, 0, 2]
        h10 = buffers.H[gp, 1, 0]
        h11 = buffers.H[gp, 1, 1]
        h12 = buffers.H[gp, 1, 2]
        h20 = buffers.H[gp, 2, 0]
        h21 = buffers.H[gp, 2, 1]
        h22 = buffers.H[gp, 2, 2]
        h30 = buffers.H[gp, 3, 0]
        h31 = buffers.H[gp, 3, 1]
        h32 = buffers.H[gp, 3, 2]
        h40 = buffers.H[gp, 4, 0]
        h41 = buffers.H[gp, 4, 1]
        h42 = buffers.H[gp, 4, 2]
        h50 = buffers.H[gp, 5, 0]
        h51 = buffers.H[gp, 5, 1]
        h52 = buffers.H[gp, 5, 2]
        h60 = buffers.H[gp, 6, 0]
        h61 = buffers.H[gp, 6, 1]
        h62 = buffers.H[gp, 6, 2]
        h70 = buffers.H[gp, 7, 0]
        h71 = buffers.H[gp, 7, 1]
        h72 = buffers.H[gp, 7, 2]

        F00 = x0 * h00 + x1 * h10 + x2 * h20 + x3 * h30 + x4 * h40 + x5 * h50 + x6 * h60 + x7 * h70
        F01 = x0 * h01 + x1 * h11 + x2 * h21 + x3 * h31 + x4 * h41 + x5 * h51 + x6 * h61 + x7 * h71
        F02 = x0 * h02 + x1 * h12 + x2 * h22 + x3 * h32 + x4 * h42 + x5 * h52 + x6 * h62 + x7 * h72

        F10 = y0 * h00 + y1 * h10 + y2 * h20 + y3 * h30 + y4 * h40 + y5 * h50 + y6 * h60 + y7 * h70
        F11 = y0 * h01 + y1 * h11 + y2 * h21 + y3 * h31 + y4 * h41 + y5 * h51 + y6 * h61 + y7 * h71
        F12 = y0 * h02 + y1 * h12 + y2 * h22 + y3 * h32 + y4 * h42 + y5 * h52 + y6 * h62 + y7 * h72

        F20 = z0 * h00 + z1 * h10 + z2 * h20 + z3 * h30 + z4 * h40 + z5 * h50 + z6 * h60 + z7 * h70
        F21 = z0 * h01 + z1 * h11 + z2 * h21 + z3 * h31 + z4 * h41 + z5 * h51 + z6 * h61 + z7 * h71
        F22 = z0 * h02 + z1 * h12 + z2 * h22 + z3 * h32 + z4 * h42 + z5 * h52 + z6 * h62 + z7 * h72

        C00 = F00 * F00 + F10 * F10 + F20 * F20
        C11 = F01 * F01 + F11 * F11 + F21 * F21
        C22 = F02 * F02 + F12 * F12 + F22 * F22
        trace_E = wp.float64(0.5) * (C00 + C11 + C22 - wp.float64(3.0))

        FC00 = F00 * C00 + F01 * (F00 * F01 + F10 * F11 + F20 * F21) + F02 * (F00 * F02 + F10 * F12 + F20 * F22)
        FC01 = F00 * (F01 * F00 + F11 * F10 + F21 * F20) + F01 * C11 + F02 * (F01 * F02 + F11 * F12 + F21 * F22)
        FC02 = F00 * (F02 * F00 + F12 * F10 + F22 * F20) + F01 * (F02 * F01 + F12 * F11 + F22 * F21) + F02 * C22

        FC10 = F10 * C00 + F11 * (F00 * F01 + F10 * F11 + F20 * F21) + F12 * (F00 * F02 + F10 * F12 + F20 * F22)
        FC11 = F10 * (F01 * F00 + F11 * F10 + F21 * F20) + F11 * C11 + F12 * (F01 * F02 + F11 * F12 + F21 * F22)
        FC12 = F10 * (F02 * F00 + F12 * F10 + F22 * F20) + F11 * (F02 * F01 + F12 * F11 + F22 * F21) + F12 * C22

        FC20 = F20 * C00 + F21 * (F00 * F01 + F10 * F11 + F20 * F21) + F22 * (F00 * F02 + F10 * F12 + F20 * F22)
        FC21 = F20 * (F01 * F00 + F11 * F10 + F21 * F20) + F21 * C11 + F22 * (F01 * F02 + F11 * F12 + F21 * F22)
        FC22 = F20 * (F02 * F00 + F12 * F10 + F22 * F20) + F21 * (F02 * F01 + F12 * F11 + F22 * F21) + F22 * C22

        P00 = lam * trace_E * F00 + mu * (FC00 - F00)
        P01 = lam * trace_E * F01 + mu * (FC01 - F01)
        P02 = lam * trace_E * F02 + mu * (FC02 - F02)
        P10 = lam * trace_E * F10 + mu * (FC10 - F10)
        P11 = lam * trace_E * F11 + mu * (FC11 - F11)
        P12 = lam * trace_E * F12 + mu * (FC12 - F12)
        P20 = lam * trace_E * F20 + mu * (FC20 - F20)
        P21 = lam * trace_E * F21 + mu * (FC21 - F21)
        P22 = lam * trace_E * F22 + mu * (FC22 - F22)

        w = buffers.weights[gp]

        pf0x = (P00 * h00 + P01 * h01 + P02 * h02) * w
        pf0y = (P10 * h00 + P11 * h01 + P12 * h02) * w
        pf0z = (P20 * h00 + P21 * h01 + P22 * h02) * w
        pf1x = (P00 * h10 + P01 * h11 + P02 * h12) * w
        pf1y = (P10 * h10 + P11 * h11 + P12 * h12) * w
        pf1z = (P20 * h10 + P21 * h11 + P22 * h12) * w
        pf2x = (P00 * h20 + P01 * h21 + P02 * h22) * w
        pf2y = (P10 * h20 + P11 * h21 + P12 * h22) * w
        pf2z = (P20 * h20 + P21 * h21 + P22 * h22) * w
        pf3x = (P00 * h30 + P01 * h31 + P02 * h32) * w
        pf3y = (P10 * h30 + P11 * h31 + P12 * h32) * w
        pf3z = (P20 * h30 + P21 * h31 + P22 * h32) * w
        pf4x = (P00 * h40 + P01 * h41 + P02 * h42) * w
        pf4y = (P10 * h40 + P11 * h41 + P12 * h42) * w
        pf4z = (P20 * h40 + P21 * h41 + P22 * h42) * w
        pf5x = (P00 * h50 + P01 * h51 + P02 * h52) * w
        pf5y = (P10 * h50 + P11 * h51 + P12 * h52) * w
        pf5z = (P20 * h50 + P21 * h51 + P22 * h52) * w
        pf6x = (P00 * h60 + P01 * h61 + P02 * h62) * w
        pf6y = (P10 * h60 + P11 * h61 + P12 * h62) * w
        pf6z = (P20 * h60 + P21 * h61 + P22 * h62) * w
        pf7x = (P00 * h70 + P01 * h71 + P02 * h72) * w
        pf7y = (P10 * h70 + P11 * h71 + P12 * h72) * w
        pf7z = (P20 * h70 + P21 * h71 + P22 * h72) * w

        f0x += pf0x
        f0y += pf0y
        f0z += pf0z
        f1x += pf1x
        f1y += pf1y
        f1z += pf1z
        f2x += pf2x
        f2y += pf2y
        f2z += pf2z
        f3x += pf3x
        f3y += pf3y
        f3z += pf3z
        f4x += pf4x
        f4y += pf4y
        f4z += pf4z
        f5x += pf5x
        f5y += pf5y
        f5z += pf5z
        f6x += pf6x
        f6y += pf6y
        f6z += pf6z
        f7x += pf7x
        f7y += pf7y
        f7z += pf7z
    wp.atomic_add(buffers.R, 3 * conn0 + 0, f0x)
    wp.atomic_add(buffers.R, 3 * conn0 + 1, f0y)
    wp.atomic_add(buffers.R, 3 * conn0 + 2, f0z)
    wp.atomic_add(buffers.R, 3 * conn1 + 0, f1x)
    wp.atomic_add(buffers.R, 3 * conn1 + 1, f1y)
    wp.atomic_add(buffers.R, 3 * conn1 + 2, f1z)
    wp.atomic_add(buffers.R, 3 * conn2 + 0, f2x)
    wp.atomic_add(buffers.R, 3 * conn2 + 1, f2y)
    wp.atomic_add(buffers.R, 3 * conn2 + 2, f2z)
    wp.atomic_add(buffers.R, 3 * conn3 + 0, f3x)
    wp.atomic_add(buffers.R, 3 * conn3 + 1, f3y)
    wp.atomic_add(buffers.R, 3 * conn3 + 2, f3z)
    wp.atomic_add(buffers.R, 3 * conn4 + 0, f4x)
    wp.atomic_add(buffers.R, 3 * conn4 + 1, f4y)
    wp.atomic_add(buffers.R, 3 * conn4 + 2, f4z)
    wp.atomic_add(buffers.R, 3 * conn5 + 0, f5x)
    wp.atomic_add(buffers.R, 3 * conn5 + 1, f5y)
    wp.atomic_add(buffers.R, 3 * conn5 + 2, f5z)
    wp.atomic_add(buffers.R, 3 * conn6 + 0, f6x)
    wp.atomic_add(buffers.R, 3 * conn6 + 1, f6y)
    wp.atomic_add(buffers.R, 3 * conn6 + 2, f6z)
    wp.atomic_add(buffers.R, 3 * conn7 + 0, f7x)
    wp.atomic_add(buffers.R, 3 * conn7 + 1, f7y)
    wp.atomic_add(buffers.R, 3 * conn7 + 2, f7z)


@wp.kernel
def assemble_tangent_kernel(
    buffers: DenseWarpBuffers,
    n_local: int,
    n_gauss: int,
    scale: wp.float64,
    lam: wp.float64,
    mu: wp.float64,
):
    # Keep the 3x3 block assembly in scalar form for the same reason as force:
    # predictable codegen and no extra local tensor staging in the dense-first path.
    e = wp.tid()

    for i in range(n_local):
        row_coef = buffers.elem_conn[e, i]
        row = 3 * row_coef

        for j in range(n_local):
            col_coef = buffers.elem_conn[e, j]
            col = 3 * col_coef

            k00 = wp.float64(0.0)
            k01 = wp.float64(0.0)
            k02 = wp.float64(0.0)
            k10 = wp.float64(0.0)
            k11 = wp.float64(0.0)
            k12 = wp.float64(0.0)
            k20 = wp.float64(0.0)
            k21 = wp.float64(0.0)
            k22 = wp.float64(0.0)

            for gp in range(n_gauss):
                F00 = wp.float64(0.0)
                F01 = wp.float64(0.0)
                F02 = wp.float64(0.0)
                F10 = wp.float64(0.0)
                F11 = wp.float64(0.0)
                F12 = wp.float64(0.0)
                F20 = wp.float64(0.0)
                F21 = wp.float64(0.0)
                F22 = wp.float64(0.0)

                for a in range(n_local):
                    coef = buffers.elem_conn[e, a]
                    h0 = buffers.H[gp, a, 0]
                    h1 = buffers.H[gp, a, 1]
                    h2 = buffers.H[gp, a, 2]

                    xa = buffers.x[coef]
                    ya = buffers.y[coef]
                    za = buffers.z[coef]

                    F00 += xa * h0
                    F01 += xa * h1
                    F02 += xa * h2
                    F10 += ya * h0
                    F11 += ya * h1
                    F12 += ya * h2
                    F20 += za * h0
                    F21 += za * h1
                    F22 += za * h2

                C00 = F00 * F00 + F10 * F10 + F20 * F20
                C11 = F01 * F01 + F11 * F11 + F21 * F21
                C22 = F02 * F02 + F12 * F12 + F22 * F22
                C01 = F00 * F01 + F10 * F11 + F20 * F21
                C02 = F00 * F02 + F10 * F12 + F20 * F22
                C12 = F01 * F02 + F11 * F12 + F21 * F22

                trE = wp.float64(0.5) * (C00 + C11 + C22 - wp.float64(3.0))

                E00 = wp.float64(0.5) * (C00 - wp.float64(1.0))
                E11 = wp.float64(0.5) * (C11 - wp.float64(1.0))
                E22 = wp.float64(0.5) * (C22 - wp.float64(1.0))
                E01 = wp.float64(0.5) * C01
                E02 = wp.float64(0.5) * C02
                E12 = wp.float64(0.5) * C12

                S00 = lam * trE + wp.float64(2.0) * mu * E00
                S11 = lam * trE + wp.float64(2.0) * mu * E11
                S22 = lam * trE + wp.float64(2.0) * mu * E22
                S01 = wp.float64(2.0) * mu * E01
                S02 = wp.float64(2.0) * mu * E02
                S12 = wp.float64(2.0) * mu * E12

                B00 = F00 * F00 + F01 * F01 + F02 * F02
                B01 = F00 * F10 + F01 * F11 + F02 * F12
                B02 = F00 * F20 + F01 * F21 + F02 * F22
                B11 = F10 * F10 + F11 * F11 + F12 * F12
                B12 = F10 * F20 + F11 * F21 + F12 * F22
                B22 = F20 * F20 + F21 * F21 + F22 * F22

                gi0 = buffers.H[gp, i, 0]
                gi1 = buffers.H[gp, i, 1]
                gi2 = buffers.H[gp, i, 2]
                gj0 = buffers.H[gp, j, 0]
                gj1 = buffers.H[gp, j, 1]
                gj2 = buffers.H[gp, j, 2]

                dot_h = gi0 * gj0 + gi1 * gj1 + gi2 * gj2

                ui0 = F00 * gi0 + F01 * gi1 + F02 * gi2
                ui1 = F10 * gi0 + F11 * gi1 + F12 * gi2
                ui2 = F20 * gi0 + F21 * gi1 + F22 * gi2

                uj0 = F00 * gj0 + F01 * gj1 + F02 * gj2
                uj1 = F10 * gj0 + F11 * gj1 + F12 * gj2
                uj2 = F20 * gj0 + F21 * gj1 + F22 * gj2

                sgj0 = S00 * gj0 + S01 * gj1 + S02 * gj2
                sgj1 = S01 * gj0 + S11 * gj1 + S12 * gj2
                sgj2 = S02 * gj0 + S12 * gj1 + S22 * gj2
                hsh = gi0 * sgj0 + gi1 * sgj1 + gi2 * sgj2

                w = buffers.weights[gp]

                k00 += w * (hsh + lam * ui0 * uj0 + mu * uj0 * ui0 + mu * B00 * dot_h)
                k01 += w * (lam * ui0 * uj1 + mu * uj0 * ui1 + mu * B01 * dot_h)
                k02 += w * (lam * ui0 * uj2 + mu * uj0 * ui2 + mu * B02 * dot_h)

                k10 += w * (lam * ui1 * uj0 + mu * uj1 * ui0 + mu * B01 * dot_h)
                k11 += w * (hsh + lam * ui1 * uj1 + mu * uj1 * ui1 + mu * B11 * dot_h)
                k12 += w * (lam * ui1 * uj2 + mu * uj1 * ui2 + mu * B12 * dot_h)

                k20 += w * (lam * ui2 * uj0 + mu * uj2 * ui0 + mu * B02 * dot_h)
                k21 += w * (lam * ui2 * uj1 + mu * uj2 * ui1 + mu * B12 * dot_h)
                k22 += w * (hsh + lam * ui2 * uj2 + mu * uj2 * ui2 + mu * B22 * dot_h)

            wp.atomic_add(buffers.J, row + 0, col + 0, scale * k00)
            wp.atomic_add(buffers.J, row + 0, col + 1, scale * k01)
            wp.atomic_add(buffers.J, row + 0, col + 2, scale * k02)
            wp.atomic_add(buffers.J, row + 1, col + 0, scale * k10)
            wp.atomic_add(buffers.J, row + 1, col + 1, scale * k11)
            wp.atomic_add(buffers.J, row + 1, col + 2, scale * k12)
            wp.atomic_add(buffers.J, row + 2, col + 0, scale * k20)
            wp.atomic_add(buffers.J, row + 2, col + 1, scale * k21)
            wp.atomic_add(buffers.J, row + 2, col + 2, scale * k22)


# -----------------------------------------------------------------------------
# Host precompute
# -----------------------------------------------------------------------------


def precompute_dense_system(
    n_elements: int = 1, l_total: float = 0.5, g_vec: np.ndarray | None = None
) -> DensePrecompute:
    """Build runtime precomputations for the dense Warp example."""
    if g_vec is None:
        g_vec = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    else:
        g_vec = np.asarray(g_vec, dtype=np.float64)

    ctx, x0, y0, z0, vx0, vy0, vz0 = build_mesh(n_elements, l_total)
    j_mass = precompute_mass_matrix(ctx)
    r_gravity = precompute_gravity_force(ctx, g_vec)
    tip_shape8 = shape_functions(ctx.L_elem / 2.0, 0.0, 0.0, ctx.B_inv)

    return DensePrecompute(
        ctx=ctx,
        x0=x0,
        y0=y0,
        z0=z0,
        vx0=vx0,
        vy0=vy0,
        vz0=vz0,
        j_mass=j_mass,
        r_gravity=r_gravity,
        tip_shape8=tip_shape8,
    )


# -----------------------------------------------------------------------------
# Backend interop and device allocation
# -----------------------------------------------------------------------------


def require_cupy() -> None:
    """Raise a clear error when the CUDA dense solve path is requested without CuPy."""
    if cp is None:
        raise RuntimeError("CuPy is required for CUDA dense solves in example_ancf_beam_dense.py")


def warp_array_to_cupy(array: wp.array):
    """Create a CuPy view of a CUDA Warp array without a host copy."""
    require_cupy()
    device = array.device
    if not device.is_cuda:
        raise ValueError(f"Expected a CUDA Warp array, got device={device}")

    with cp.cuda.Device(device.ordinal):
        return cp.asarray(array)


def build_warp_state(pre: DensePrecompute, device: str | None = "cpu") -> DenseWarpState:
    """Allocate the first-step Warp state buffers on the selected device."""
    warp_device = wp.get_device(device)
    n_coef = pre.ctx.N_coef
    n_dofs = pre.ctx.n_dofs

    buffers = DenseWarpBuffers()
    buffers.n_coef = n_coef
    buffers.n_dofs = n_dofs
    buffers.elem_conn = wp.array2d(pre.ctx.elem_conn.astype(np.int32), dtype=wp.int32, device=warp_device)
    buffers.weights = wp.array(pre.ctx.gauss.weights, dtype=wp.float64, device=warp_device)
    buffers.H = wp.array3d(pre.ctx.gauss.H, dtype=wp.float64, device=warp_device)
    buffers.j_mass = wp.array2d(pre.j_mass, dtype=wp.float64, device=warp_device)
    buffers.r_gravity = wp.array(pre.r_gravity, dtype=wp.float64, device=warp_device)
    buffers.tip_shape8 = wp.array(pre.tip_shape8, dtype=wp.float64, device=warp_device)
    buffers.x = wp.array(pre.x0, dtype=wp.float64, device=warp_device)
    buffers.y = wp.array(pre.y0, dtype=wp.float64, device=warp_device)
    buffers.z = wp.array(pre.z0, dtype=wp.float64, device=warp_device)
    buffers.vx = wp.array(pre.vx0, dtype=wp.float64, device=warp_device)
    buffers.vy = wp.array(pre.vy0, dtype=wp.float64, device=warp_device)
    buffers.vz = wp.array(pre.vz0, dtype=wp.float64, device=warp_device)
    buffers.ax = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.ay = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.az = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.x_prev = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.y_prev = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.z_prev = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.vx_prev = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.vy_prev = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.vz_prev = wp.zeros(n_coef, dtype=wp.float64, device=warp_device)
    buffers.J = wp.zeros((n_dofs, n_dofs), dtype=wp.float64, device=warp_device)
    buffers.R = wp.zeros(n_dofs, dtype=wp.float64, device=warp_device)
    buffers.delta_a = wp.zeros(n_dofs, dtype=wp.float64, device=warp_device)

    return DenseWarpState(device=warp_device, buffers=buffers)


# -----------------------------------------------------------------------------
# Dense assembly and Newton solve
# -----------------------------------------------------------------------------


def solve_dense_system(state: DenseWarpState) -> None:
    """Solve the current dense system on the active backend and store the result in ``delta_a``."""
    if state.device.is_cuda:
        J_cu = warp_array_to_cupy(state.buffers.J)
        R_cu = warp_array_to_cupy(state.buffers.R)
        delta_cu = warp_array_to_cupy(state.buffers.delta_a)
        delta_cu[...] = cp.linalg.solve(J_cu, -R_cu)
        return

    delta = np.linalg.solve(state.buffers.J.numpy(), -state.buffers.R.numpy())
    wp.copy(dest=state.buffers.delta_a, src=wp.array(delta, dtype=wp.float64, device=state.device))


def residual_norm(state: DenseWarpState, n_constraints: int) -> float:
    """Return the unconstrained residual norm on the active backend."""
    if state.device.is_cuda:
        R_cu = warp_array_to_cupy(state.buffers.R)
        return float(cp.linalg.norm(R_cu[n_constraints:]).get())

    return float(np.linalg.norm(state.buffers.R.numpy()[n_constraints:]))


def evaluate_residual(state: DenseWarpState, pre: DensePrecompute, h: float, t: float) -> None:
    """Evaluate the clamped Newton residual for the current acceleration iterate."""
    wp.launch(
        kernel=compute_trial_position_kernel,
        dim=state.n_coef,
        inputs=[state.buffers, float(h)],
        device=state.device,
    )
    wp.launch(
        kernel=initialize_residual_kernel,
        dim=state.n_dofs,
        inputs=[state.buffers],
        device=state.device,
    )
    wp.launch(
        kernel=assemble_internal_force_kernel,
        dim=pre.ctx.n_beam,
        inputs=[state.buffers, pre.ctx.lam, pre.ctx.mu],
        device=state.device,
    )
    tip_force = F_tip(t)
    wp.launch(
        kernel=apply_tip_load_kernel,
        dim=8,
        inputs=[
            state.buffers,
            pre.ctx.n_beam - 1,
            float(tip_force[0]),
            float(tip_force[1]),
            float(tip_force[2]),
        ],
        device=state.device,
    )
    wp.launch(kernel=clamp_residual_kernel, dim=pre.ctx.n_constraints, inputs=[state.buffers], device=state.device)


def assemble_newton_matrix(state: DenseWarpState, pre: DensePrecompute, h: float) -> None:
    """Assemble the dense Newton matrix after the residual check requires a solve."""
    wp.copy(dest=state.buffers.J, src=state.buffers.j_mass)
    wp.launch(
        kernel=assemble_tangent_kernel,
        dim=pre.ctx.n_beam,
        inputs=[
            state.buffers,
            8,
            len(pre.ctx.gauss.weights),
            float(h * h),
            pre.ctx.lam,
            pre.ctx.mu,
        ],
        device=state.device,
    )
    wp.launch(kernel=clamp_columns_kernel, dim=pre.ctx.n_constraints, inputs=[state.buffers], device=state.device)
    wp.launch(
        kernel=clamp_matrix_rows_kernel,
        dim=pre.ctx.n_constraints,
        inputs=[state.buffers],
        device=state.device,
    )


def solve_one_step_newton(
    state: DenseWarpState,
    pre: DensePrecompute,
    t: float,
    h: float,
    tol: float = 1.0e-6,
    max_iter: int = 10,
) -> None:
    """Solve one clamped Newton step in-place on an existing Warp state."""
    wp.launch(kernel=save_prev_state_kernel, dim=state.n_coef, inputs=[state.buffers], device=state.device)

    for _ in range(max_iter):
        evaluate_residual(state, pre, h, t)
        if residual_norm(state, pre.ctx.n_constraints) < tol:
            break

        assemble_newton_matrix(state, pre, h)
        solve_dense_system(state)
        wp.launch(
            kernel=update_acceleration_from_delta_kernel,
            dim=state.n_coef,
            inputs=[state.buffers],
            device=state.device,
        )

    wp.launch(
        kernel=compute_trial_position_kernel,
        dim=state.n_coef,
        inputs=[state.buffers, float(h)],
        device=state.device,
    )


# -----------------------------------------------------------------------------
# Time integration and output
# -----------------------------------------------------------------------------


def simulate_bdf1_warp(
    pre: DensePrecompute,
    t0: float,
    tf: float,
    h: float,
    tol: float = 1.0e-6,
    max_iter: int = 10,
    device: str = "cpu",
    write_output: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run the Warp-assembled dense path and optionally record tip history."""
    state = build_warp_state(pre, device=device)

    n_steps = int((tf - t0) / h)
    tip_history = None
    saved_times = None

    if write_output:
        saved_times = t0 + h * np.arange(n_steps + 1, dtype=np.float64)
        tip_history = wp.empty((n_steps + 1, 3), dtype=wp.float64, device=state.device)
        wp.launch(
            kernel=record_tip_kernel,
            dim=1,
            inputs=[state.buffers, tip_history, 0, pre.ctx.n_beam - 1],
            device=state.device,
        )

    for step in range(1, n_steps + 1):
        t = t0 + step * h
        solve_one_step_newton(state, pre, t, h, tol=tol, max_iter=max_iter)
        wp.launch(kernel=accept_step_kernel, dim=state.n_coef, inputs=[state.buffers, float(h)], device=state.device)

        if tip_history is not None:
            wp.launch(
                kernel=record_tip_kernel,
                dim=1,
                inputs=[state.buffers, tip_history, step, pre.ctx.n_beam - 1],
                device=state.device,
            )

    if tip_history is None:
        return None, None

    return tip_history.numpy(), saved_times


def save_tip_positions_csv(
    tip_history: np.ndarray,
    saved_times: np.ndarray,
    h: float,
    n_beam: int,
    output_dir: str | None = None,
) -> str:
    """Write the tip trajectory CSV using the reference-compatible file format."""
    print("\nWriting tip positions to CSV...")
    fz = np.asarray([F_tip(t)[2] for t in saved_times], dtype=np.float64)
    data = np.column_stack([saved_times, tip_history[:, 0], tip_history[:, 1], tip_history[:, 2], fz])

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tip_positions_refined_{n_beam}_h{h:.0e}.csv")
    header = "time(s),tip_x(m),tip_y(m),tip_z(m),force_z(N)"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="", fmt="%.10e")
    print(f"CSV saved: {output_path}")
    return output_path


def main(
    n_elements: int = 1,
    tf: float = 10.0,
    h: float = 5.0e-4,
    device: str = "cpu",
    output_dir: str | None = None,
    tol: float = 1.0e-6,
    max_iter: int = 10,
    write_output: bool = True,
) -> None:
    """Run the dense Warp ANCF example."""
    wp.init()

    pre = precompute_dense_system(n_elements=n_elements)

    print("ANCF dense Warp example")
    print(f"  Warp device: {wp.get_device(device)}")
    print(f"  Elements: {pre.ctx.n_beam}")
    print(f"  DOFs: {pre.ctx.n_dofs}")
    print(f"  tf: {tf}")
    print(f"  h: {h}")

    print("\nRunning BDF-1 integration with clamped BC:")
    print(f"  Time: {0.0:.6f} to {tf:.6f} s")
    print(f"  Time step: {h:.6e} s")
    print(f"  Number of steps: {int((tf - 0.0) / h)}")
    output_text = "every step" if write_output else "disabled"
    print(f"  Output: {output_text}")
    print(f"  Newton tolerance: {tol:.2e}")
    print(f"  Max Newton iterations: {max_iter}")
    print(f"  Clamped DOFs 0..{pre.ctx.n_constraints - 1} fixed at {CLAMP_TARGET}\n")

    tip_history, saved_times = simulate_bdf1_warp(
        pre,
        t0=0.0,
        tf=tf,
        h=h,
        tol=tol,
        max_iter=max_iter,
        device=device,
        write_output=write_output,
    )

    print("\nIntegration complete!")
    print(f"Final time: {tf:.6f} s")
    if tip_history is not None and saved_times is not None:
        save_tip_positions_csv(
            tip_history,
            saved_times,
            h,
            n_beam=pre.ctx.n_beam,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-elements", type=int, default=1)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--h", type=float, default=5.0e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args()

    main(
        n_elements=args.n_elements,
        tf=args.tf,
        h=args.h,
        device=args.device,
        output_dir=args.output_dir,
        tol=args.tol,
        max_iter=args.max_iter,
        write_output=not args.no_output,
    )
