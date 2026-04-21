"""
ANCF flexible-beam multibody-dynamics simulator (NumPy, CPU).

Cantilever beam clamped at the origin, with a short time-varying tip load
plus gravity, integrated in time with BDF-1 (implicit Euler) and a Lagrange
multiplier constraint on the clamped node. St. Venant-Kirchhoff material.

Element: B3-24 ANCF (8 nodes x 3 DOFs = 24 DOFs per element).
Each element has 8 "nodes" grouped as [position, d/du, d/dv, d/dw] at each
of the two end points. Adjacent elements share 4 nodes (right end of elem e
== left end of elem e+1).

DOF layout (load-bearing invariant):
    x_n is (3 * N_coef,) float64
    x_n[3*k + 0:3*k+3] is (x, y, z) of global node k, for k in [0, N_coef)
    N_coef = 8 + 4 * (n_beam - 1)

Connectivity:
    elem_conn: (n_beam, 8) int64
    elem_conn[e, i_local] = 4*e + i_local   (global node index)
    Rows overlap by 4 between adjacent elements.
"""

import os
from dataclasses import dataclass

import numpy as np

# Tip-trajectory CSVs go to the repo-level scratch directory (gitignored), so
# this example doesn't leave build artifacts in warp/examples/fem/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "temp", "ancf", "output")


# =========================================================================
# Gauss-Legendre quadrature
# =========================================================================

GL_TABLE = {
    2: {
        "x": np.array([-0.5773502691896257, 0.5773502691896257]),
        "w": np.array([1.0000000000000000, 1.0000000000000000]),
    },
    3: {
        "x": np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
        "w": np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
    },
    4: {
        "x": np.array([-0.3399810435848563, 0.3399810435848563, -0.8611363115940526, 0.8611363115940526]),
        "w": np.array([0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0.3478548451374538]),
    },
    5: {
        "x": np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]),
        "w": np.array(
            [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
        ),
    },
}


def gauss_legendre(n):
    data = GL_TABLE[n]
    return data["x"], data["w"]


# =========================================================================
# ANCF B3-24 basis
# =========================================================================


def eval_basis(u, v, w):
    return np.array([1.0, u, v, w, u * v, u * w, u**2, u**3])


def eval_basis_derivatives(u, v, w):
    ddu = np.array([0.0, 1.0, 0.0, 0.0, v, w, 2 * u, 3 * u**2])
    ddv = np.array([0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0, 0.0])
    ddw = np.array([0.0, 0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0])
    return np.array([ddu, ddv, ddw])  # (3, 8)


def build_B_matrix(L_elem):
    B = np.zeros((8, 8))

    u1, v1, w1 = -L_elem / 2, 0.0, 0.0
    basis1 = eval_basis(u1, v1, w1)
    basis_derivs1 = eval_basis_derivatives(u1, v1, w1)
    B[0, :] = basis1
    B[1, :] = basis_derivs1[0, :]
    B[2, :] = basis_derivs1[1, :]
    B[3, :] = basis_derivs1[2, :]

    u2, v2, w2 = L_elem / 2, 0.0, 0.0
    basis2 = eval_basis(u2, v2, w2)
    basis_derivs2 = eval_basis_derivatives(u2, v2, w2)
    B[4, :] = basis2
    B[5, :] = basis_derivs2[0, :]
    B[6, :] = basis_derivs2[1, :]
    B[7, :] = basis_derivs2[2, :]

    return B


def shape_functions(u, v, w, B_inv):
    """Return the 8 nodal shape functions at (u, v, w)."""
    return B_inv @ eval_basis(u, v, w)


# =========================================================================
# Context
# =========================================================================


@dataclass(frozen=True)
class GaussData:
    weights: np.ndarray  # (16,) float64
    S: np.ndarray  # (16, 8) float64 - shape-fn values at each gp
    H: np.ndarray  # (16, 8, 3) float64 - shape-fn gradients at each gp


@dataclass(frozen=True)
class SimContext:
    # Mesh sizes
    n_beam: int
    N_coef: int  # 8 + 4*(n_beam - 1)
    n_dofs: int  # 3 * N_coef
    n_constraints: int  # always 12 (clamped cantilever)

    # Geometry / material
    L_elem: float
    W: float
    H: float
    rho: float
    E: float
    nu: float
    lam: float
    mu: float

    # Connectivity
    elem_conn: np.ndarray  # (n_beam, 8) int64

    # Element basis
    B_inv: np.ndarray  # (8, 8) float64
    gauss: GaussData


def _build_B_inv(L_elem):
    return np.linalg.inv(build_B_matrix(L_elem).T)


def _build_gauss(L_elem, W, H, B_inv):
    J_det = (L_elem * W * H) / 8.0

    gp_u, w_u = gauss_legendre(4)
    gp_v, w_v = gauss_legendre(2)
    gp_w, w_w = gauss_legendre(2)

    weights_list = []
    S_list = []
    H_list = []

    for i_u, xi in enumerate(gp_u):
        u = (L_elem / 2) * xi
        for i_v, eta in enumerate(gp_v):
            v = (W / 2) * eta
            for i_w, zeta in enumerate(gp_w):
                w = (H / 2) * zeta
                weights_list.append(w_u[i_u] * w_v[i_v] * w_w[i_w] * J_det)
                basis = eval_basis(u, v, w)
                basis_derivs = eval_basis_derivatives(u, v, w)
                S_list.append(B_inv @ basis)
                H_list.append(B_inv @ basis_derivs.T)

    return GaussData(
        weights=np.array(weights_list),
        S=np.array(S_list),
        H=np.array(H_list),
    )


def build_mesh(n_elements, L_total, W=0.003, H=0.003, rho=7700.0, E=2.0e11, nu=0.3):
    """Build a straight cantilever beam along +x from origin.

    Returns (ctx, x0, v0) where x0/v0 are (n_dofs,) arrays.
    """
    L_elem = L_total / n_elements
    n_beam = n_elements
    N_coef = 8 + 4 * (n_beam - 1)
    n_dofs = 3 * N_coef

    # Connectivity: row e = [4e, 4e+1, ..., 4e+7], rows overlap by 4.
    elem_conn = 4 * np.arange(n_beam, dtype=np.int64)[:, None] + np.arange(8, dtype=np.int64)[None, :]

    B_inv = _build_B_inv(L_elem)
    gauss = _build_gauss(L_elem, W, H, B_inv)

    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    ctx = SimContext(
        n_beam=n_beam,
        N_coef=N_coef,
        n_dofs=n_dofs,
        n_constraints=12,
        L_elem=L_elem,
        W=W,
        H=H,
        rho=rho,
        E=E,
        nu=nu,
        lam=lam,
        mu=mu,
        elem_conn=elem_conn,
        B_inv=B_inv,
        gauss=gauss,
    )

    # Closed-form straight-beam initial state.
    # For global node k: let g = k // 4 (group index), r = k % 4 (role:
    # 0=position, 1=d/du, 2=d/dv, 3=d/dw).
    x0 = np.zeros(n_dofs)
    k = np.arange(N_coef)
    g = k // 4
    r = k % 4
    x0[3 * k + 0] = np.where(r == 0, g * L_elem, np.where(r == 1, 1.0, 0.0))
    x0[3 * k + 1] = np.where(r == 2, 1.0, 0.0)
    x0[3 * k + 2] = np.where(r == 3, 1.0, 0.0)
    v0 = np.zeros(n_dofs)

    print(f"Element Initialized: Length={L_elem:.4f}m, J_det={(L_elem * W * H) / 8.0:.2e}")
    return ctx, x0, v0


# =========================================================================
# External load
# =========================================================================


def F_tip(t):
    """Time-dependent tip load, applied on [0, 0.05] s. Inclusive endpoints."""
    if 0.0 <= t <= 0.05:
        force_z = -1.0 + np.cos(20.0 * np.pi * t)
        return np.array([0.0, 0.0, force_z])
    else:
        return np.array([0.0, 0.0, 0.0])


# =========================================================================
# Constant matrices: mass, gravity
# =========================================================================


def compute_mass_matrix(ctx):
    """Global consistent mass matrix, (n_dofs, n_dofs)."""
    S = ctx.gauss.S  # (16, 8)
    weights = ctx.gauss.weights  # (16,)

    m_global = np.zeros((ctx.N_coef, ctx.N_coef))

    for elem in range(ctx.n_beam):
        m_elem = np.zeros((8, 8))
        for gp_idx in range(16):
            s = S[gp_idx, :]
            weight = weights[gp_idx]
            m_elem += ctx.rho * np.outer(s, s) * weight

        m_elem = 0.5 * (m_elem + m_elem.T)  # symmetrize (matches original)

        conn = ctx.elem_conn[elem]
        m_global[np.ix_(conn, conn)] += m_elem

    I3 = np.eye(3)
    return np.kron(m_global, I3)


def compute_gravity_force(ctx, g_vec):
    """Global gravity force vector, (n_dofs, 1)."""
    S = ctx.gauss.S
    weights = ctx.gauss.weights

    G_global = np.zeros((ctx.N_coef, 3))

    for elem in range(ctx.n_beam):
        # Explicit sum to match original FP order (NumPy pairwise would differ).
        v_i = np.zeros(8)
        for gp_idx in range(16):
            s = S[gp_idx, :]
            weight = weights[gp_idx]
            v_i += s * weight

        G_elem = ctx.rho * np.outer(v_i, g_vec)  # (8, 3)

        conn = ctx.elem_conn[elem]
        G_global[conn, :] += G_elem

    return G_global.reshape(-1, 1)


# =========================================================================
# Internal force (St. Venant-Kirchhoff)
# =========================================================================


def compute_internal_force(ctx, x_n):
    """Global internal force vector f_int, (n_dofs, 1).

    SVK stress P = lam * tr(E) * F + mu * (F F^T F - F),
    i.e. P = F * (lam * tr(E) * I + 2 mu * E) with the 2*mu distributed into
    (F^T F - I). Numerically identical to the standard SVK form.
    """
    H = ctx.gauss.H  # (16, 8, 3)
    weights = ctx.gauss.weights  # (16,)
    lam, mu = ctx.lam, ctx.mu
    I3 = np.eye(3)

    f_int_global = np.zeros((ctx.n_dofs, 1))
    # View into the same buffer as (N_coef, 3) for the scatter.
    f_int_nodes = f_int_global.reshape(ctx.N_coef, 3)

    # Nodes-first view of current positions: shape (N_coef, 3).
    x_nodes = x_n.reshape(ctx.N_coef, 3)

    for elem in range(ctx.n_beam):
        conn = ctx.elem_conn[elem]  # (8,)

        # Gather local nodal positions as (3, 8) Nmat (matches original layout).
        Nmat = x_nodes[conn, :].T  # (3, 8)

        # Gauss-point kernel — identical einsum chain to the original.
        F_all = np.einsum("ij,gjk->gik", Nmat, H)  # (16, 3, 3)
        E_all = 0.5 * (np.einsum("gji,gjk->gik", F_all, F_all) - I3[None, :, :])
        trace_E_all = np.trace(E_all, axis1=1, axis2=2)  # (16,)
        F_FT_F_all = np.einsum("gil,gjl,gjk->gik", F_all, F_all, F_all)
        P_all = lam * trace_E_all[:, None, None] * F_all + mu * (F_FT_F_all - F_all)  # (16, 3, 3)
        f_int_elem = np.einsum("gij,gkj,g->ik", H, P_all, weights)  # (8, 3)

        # Scatter (conn has 8 distinct indices within one element — safe).
        f_int_nodes[conn, :] += f_int_elem

    return f_int_global


# =========================================================================
# Constraints: node 1 clamped (position + 3 gradient frames)
# =========================================================================

# Clamped-cantilever target for the first 12 DOFs of node 1:
# position at origin with identity gradient frame (tangent along +x, normals
# along +y, +z). Used in compute_constraint and in informational prints.
CLAMP_TARGET = np.array(
    [
        0.0,
        0.0,
        0.0,  # position at origin
        1.0,
        0.0,
        0.0,  # d/du (tangent along +x)
        0.0,
        1.0,
        0.0,  # d/dv
        0.0,
        0.0,
        1.0,  # d/dw
    ]
)


def compute_constraint(x_n):
    """c = x_first_12 - CLAMP_TARGET, (12, 1). Zero when clamped."""
    return (x_n[0:12] - CLAMP_TARGET).reshape(-1, 1)


def compute_constraint_derivative(ctx):
    """c_e = d(c)/d(x_n), (12, n_dofs). Identity on first 12 DOFs."""
    c_e = np.zeros((12, ctx.n_dofs))
    c_e[0:12, 0:12] = np.eye(12)
    return c_e


# =========================================================================
# Tangent stiffness (finite-difference Jacobian of f_int)
# =========================================================================


def compute_tangent_stiffness_numerical(ctx, x_n, eps=1e-6):
    """Numerical tangent K[i, j] = (f_int(x + eps*e_j) - f_int(x))[i] / eps.

    Column-serial loop — O(n_dofs) extra force evals per call. Kept as a
    validation oracle for the analytic tangent; not used by the solver path.
    """
    n = ctx.n_dofs
    f0 = compute_internal_force(ctx, x_n).ravel()

    K = np.zeros((n, n))
    for j in range(n):
        xp = x_n.copy()
        xp[j] += eps
        fp = compute_internal_force(ctx, xp).ravel()
        K[:, j] = (fp - f0) / eps
    return K


def compute_tangent_analytic(ctx, x_n):
    """Analytic tangent K = df_int/dx for St. Venant-Kirchhoff material.

    Derived from P = F*S, S = lam*tr(E)*I + 2*mu*E, E = 0.5*(F^T F - I):

        dP_ij/dF_mn = delta_im*S_nj + lam*F_ij*F_mn
                    + mu*F_in*F_mj  + mu*B_im*delta_jn        (B = F F^T)

    Chain-ruled with dF_mn/dx[J,b] = delta_mb*H[g,J,n] gives

        K_IJ[a,b] = sum_g w_g * (
              delta_ab * (H_I . S . H_J)
            + lam * u_I[a] * u_J[b]
            + mu  * u_J[a] * u_I[b]
            + mu  * B_ab * (H_I . H_J)
        )    with u_K = F . H_K  (3-vector per node per GP).

    K is symmetric (first Piola-Kirchhoff of a hyperelastic potential).
    Single element-loop pass, fully vectorized over 16 Gauss points.
    """
    H = ctx.gauss.H  # (16, 8, 3)
    W = ctx.gauss.weights  # (16,)
    lam, mu = ctx.lam, ctx.mu
    I3 = np.eye(3)

    K_global = np.zeros((ctx.n_dofs, ctx.n_dofs))
    x_nodes = x_n.reshape(ctx.N_coef, 3)

    for elem in range(ctx.n_beam):
        conn = ctx.elem_conn[elem]  # (8,)
        Nmat = x_nodes[conn, :].T  # (3, 8)

        # Gauss-point kinematics and stress
        F_all = np.einsum("ij,gjk->gik", Nmat, H)  # (16, 3, 3)  F
        E_all = 0.5 * (np.einsum("gji,gjk->gik", F_all, F_all) - I3[None, :, :])  # (16, 3, 3)  E
        trE = np.trace(E_all, axis1=1, axis2=2)  # (16,)       tr(E)
        S_all = lam * trE[:, None, None] * I3[None, :, :] + 2.0 * mu * E_all  # (16, 3, 3)  S (2nd-PK)
        B_all = np.einsum("gan,gcn->gac", F_all, F_all)  # (16, 3, 3)  F F^T
        U_all = np.einsum("gan,gIn->gIa", F_all, H)  # (16, 8, 3)  u_I = F H_I
        Dot = np.einsum("gIn,gJn->gIJ", H, H)  # (16, 8, 8)  H_I . H_J
        HSH = np.einsum("gIn,gnc,gJc->gIJ", H, S_all, H)  # (16, 8, 8)  H_I . S . H_J

        # Element tangent block indexed (I, a, J, b) = (8, 3, 8, 3)
        T1 = np.einsum("g,gIJ,ab->IaJb", W, HSH, I3)
        T2 = lam * np.einsum("g,gIa,gJb->IaJb", W, U_all, U_all)
        T3 = mu * np.einsum("g,gJa,gIb->IaJb", W, U_all, U_all)
        T4 = mu * np.einsum("g,gab,gIJ->IaJb", W, B_all, Dot)
        K_elem = (T1 + T2 + T3 + T4).reshape(24, 24)

        # Scatter into global using flat DOF indices; conn has 8 distinct
        # entries within one element so += under fancy indexing is safe.
        rows = (3 * conn[:, None] + np.arange(3)[None, :]).ravel()
        K_global[np.ix_(rows, rows)] += K_elem

    return K_global


# =========================================================================
# Residual and Jacobian for BDF-1 + Lagrange multipliers
# =========================================================================


def compute_residual(ctx, a_n, lambda_n, x_prev, v_prev, t, h, M_e, G_f):
    """Augmented residual [R_dyn; R_con/h^2], shape (n_dofs + 12, 1)."""
    a_n = np.asarray(a_n).reshape(-1, 1)
    lambda_n = np.asarray(lambda_n).reshape(-1, 1)
    x_prev = np.asarray(x_prev).flatten()
    v_prev = np.asarray(v_prev).flatten()
    G_f = np.asarray(G_f).reshape(-1, 1)

    # BDF-1: current x from current acceleration guess.
    x_n = x_prev + h * v_prev + h**2 * a_n.flatten()

    f_int = compute_internal_force(ctx, x_n)

    # Tip load (inlined point_load_vector) — tip of last element at +L_elem/2.
    f_tip_vec = F_tip(t)
    conn_tip = ctx.elem_conn[-1]
    s_tip = shape_functions(ctx.L_elem / 2, 0.0, 0.0, ctx.B_inv)  # (8,)
    F_ext_global = np.zeros((ctx.n_dofs, 1))
    F_ext_global.reshape(ctx.N_coef, 3)[conn_tip, :] = np.outer(s_tip, f_tip_vec)

    c_e = compute_constraint_derivative(ctx)

    R_dynamics = M_e @ a_n + c_e.T @ lambda_n + f_int - G_f - F_ext_global
    R_constraint = compute_constraint(x_n) / h**2
    return np.vstack([R_dynamics, R_constraint])


def compute_jacobian(ctx, x_n, M_e, h):
    """Augmented Jacobian [[M + h^2 K, c_e^T], [c_e, 0]]."""
    K_tangent = compute_tangent_analytic(ctx, x_n)
    c_e = compute_constraint_derivative(ctx)

    n = ctx.n_dofs
    m = ctx.n_constraints
    size_J = n + m

    J = np.zeros((size_J, size_J))
    J[0:n, 0:n] = M_e + h**2 * K_tangent
    J[0:n, n:size_J] = c_e.T
    J[n:size_J, 0:n] = c_e
    # bottom-right block stays zero
    return J


# =========================================================================
# Newton-Raphson solve for acceleration + Lagrange multipliers
# =========================================================================


def solve_acceleration_newton(
    ctx, x_prev, v_prev, t, h, M_e, G_f, tol=1e-6, max_iter=10, a_init=None, lambda_init=None
):
    """Solve the augmented system for (a_n, lambda_n) at step t.

    Warm-started across time steps via a_init / lambda_init.
    """
    n = ctx.n_dofs
    m = ctx.n_constraints

    a_n = np.zeros(n) if a_init is None else a_init.copy()
    lambda_n = np.zeros(m) if lambda_init is None else lambda_init.copy()

    for _iter_num in range(max_iter):
        R = compute_residual(ctx, a_n.reshape(-1, 1), lambda_n.reshape(-1, 1), x_prev, v_prev, t, h, M_e, G_f)
        R_norm = np.linalg.norm(R)
        if R_norm < tol:
            break

        x_n = x_prev + h * v_prev + h**2 * a_n
        J = compute_jacobian(ctx, x_n, M_e, h)

        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            break

        delta_a = delta[0:n].flatten()
        delta_lambda = delta[n : n + m].flatten()

        a_n += delta_a
        lambda_n += delta_lambda

    return a_n, lambda_n


# =========================================================================
# BDF-1 time integrator
# =========================================================================


def simulate_bdf1(ctx, t0, tf, h, x0, v0, tol=1e-6, max_iter=10):
    """BDF-1 (implicit Euler) with Lagrange-multiplier clamped BC.

    Always saves every step; returns (x_final, v_final, states, times, lambdas).
    """
    print("Precomputing constant matrices...")
    M_e = compute_mass_matrix(ctx)
    g_vec = np.array([0.0, 0.0, -9.81])
    G_f = compute_gravity_force(ctx, g_vec).flatten()

    x_n = x0.copy()
    v_n = v0.copy()
    a_n = np.zeros(ctx.n_dofs)
    lambda_n = np.zeros(ctx.n_constraints)

    t = t0
    n_steps = int((tf - t0) / h)

    saved_states = [x_n.copy()]
    saved_times = [t0]
    saved_lambdas = [lambda_n.copy()]

    print("\nRunning BDF-1 integration with constraints:")
    print(f"  Time: {t0:.6f} to {tf:.6f} s")
    print(f"  Time step: {h:.6e} s")
    print(f"  Number of steps: {n_steps}")
    print(f"  Newton tolerance: {tol:.2e}")
    print(f"  Max Newton iterations: {max_iter}")
    print(f"  Constraint: Node 1 clamped (position + gradients) at {CLAMP_TARGET}\n")

    for step in range(1, n_steps + 1):
        t = t0 + step * h
        a_n, lambda_n = solve_acceleration_newton(
            ctx, x_n, v_n, t, h, M_e, G_f, tol, max_iter, a_init=a_n, lambda_init=lambda_n
        )

        x_n = x_n + h * v_n + h**2 * a_n
        v_n = v_n + h * a_n

        saved_states.append(x_n.copy())
        saved_times.append(t)
        saved_lambdas.append(lambda_n.copy())

    print("\nIntegration complete!")
    print(f"Final time: {t:.6f} s")

    return x_n, v_n, saved_states, saved_times, saved_lambdas


# =========================================================================
# Tip-position CSV output
# =========================================================================


def save_tip_positions_csv(ctx, saved_states, saved_times, h, output_dir=None):
    """Write tip trajectory CSV.

    Tip = +L_elem/2 on the last element, u-axis. File name matches the
    original for drop-in golden comparison. Defaults to DEFAULT_OUTPUT_DIR
    (repo-level temp/ancf/); override via output_dir.
    """
    print("\nComputing tip positions for CSV...")

    conn_tip = ctx.elem_conn[-1]
    s_tip = shape_functions(ctx.L_elem / 2, 0.0, 0.0, ctx.B_inv)  # (8,)

    n_frames = len(saved_states)
    tip = np.empty((n_frames, 3))
    fz = np.empty(n_frames)

    for i, state in enumerate(saved_states):
        Nmat = state.reshape(ctx.N_coef, 3)[conn_tip, :].T  # (3, 8)
        tip[i] = Nmat @ s_tip
        fz[i] = F_tip(saved_times[i])[2]

    data = np.column_stack([saved_times, tip[:, 0], tip[:, 1], tip[:, 2], fz])

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tip_positions_refined_{ctx.n_beam}_h{h:.0e}.csv")
    header = "time(s),tip_x(m),tip_y(m),tip_z(m),force_z(N)"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="", fmt="%.10e")
    print(f"CSV saved: {output_path}")


# =========================================================================
# Entry point
# =========================================================================


def main(n_elements: int = 1, tf: float = 10.0, h: float = 5e-4):
    L_total = 0.5
    ctx, x0, v0 = build_mesh(n_elements, L_total)
    print(f"SIMULATION SETUP: L_total={L_total}m, Elements={ctx.n_beam}, L_elem={ctx.L_elem:.4f}m")

    _, _, saved_states, saved_times, _ = simulate_bdf1(ctx, t0=0.0, tf=tf, h=h, x0=x0, v0=v0, tol=1e-6, max_iter=100)

    save_tip_positions_csv(ctx, saved_states, saved_times, h)


if __name__ == "__main__":
    main()
