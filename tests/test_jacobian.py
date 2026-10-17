"""The analytic sparse Jacobian must match finite differences of the residual.

Newton's quadratic convergence, and therefore every number in the reports,
rests on these derivative blocks being exact.  The realistic device is included
because band-gap narrowing splits the Bernoulli arguments per carrier species
(``de +/- d ln n_i``), which is where a copy-paste of the electron coefficients
into the hole rows would first show up.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from p5dd import DDSolver, reference_device, realistic_device

V_PROBE = 0.30
#: Central-difference step.  The residual reaches 1e8 in the space-charge
#: region, so the step is a compromise: cancellation noise scales as
#: ``|F| / (2 h)`` and truncation as ``h^2``; at h = 1e-4 both stay near 1e-9 of
#: the row scale.
H_FD = 1e-4


def _fd_matrix(sol, u, V):
    n = u.size
    out = np.empty((n, n))
    for k in range(n):
        up = u.copy()
        up[k] += H_FD
        um = u.copy()
        um[k] -= H_FD
        out[:, k] = (sol.residual(up, V) - sol.residual(um, V)) / (2.0 * H_FD)
    return out


def _row_scaled_error(Ja, Jf):
    """Element-wise error divided by the largest entry of its own row.

    The rows span some fifteen orders of magnitude (an identity Dirichlet row
    next to a continuity row whose fluxes reach 1e8), so an absolute or a
    per-entry relative tolerance would each miss half of the matrix.
    """
    Jf = np.atleast_2d(Jf)
    scale = np.maximum(np.max(np.abs(Jf), axis=1, keepdims=True), 1e-30)
    return np.abs(Ja - Jf) / scale


@pytest.fixture(scope="module")
def solved_states():
    """A converged state per material model, shared by the Jacobian checks."""
    out = {}
    for name, factory in (("reference", reference_device),
                          ("realistic", realistic_device)):
        dev = factory(n_nodes=151)
        sol = DDSolver(dev, tol=1e-6)
        u, rep = sol.solve_point(V_PROBE)
        assert rep.converged, f"{name} did not converge at V={V_PROBE}"
        out[name] = (sol, u)
    return out


@pytest.mark.parametrize("which", ["reference", "realistic"])
def test_jacobian_entries_match_finite_differences(solved_states, which):
    sol, u = solved_states[which]
    _, J = sol.residual(u, V_PROBE, want_jac=True)
    err = _row_scaled_error(J.toarray(), _fd_matrix(sol, u, V_PROBE))
    assert err.max() < 1e-6, f"worst Jacobian entry error {err.max():.3e}"


@pytest.mark.parametrize("which", ["reference", "realistic"])
def test_jacobian_action_on_a_vector(solved_states, which):
    """The matrix-free product is what Newton really depends on."""
    sol, u = solved_states[which]
    _, J = sol.residual(u, V_PROBE, want_jac=True)
    rng = np.random.default_rng(7)
    w = rng.normal(size=u.size)
    Fp = sol.residual(u + H_FD * w, V_PROBE)
    Fm = sol.residual(u - H_FD * w, V_PROBE)
    directional_fd = (Fp - Fm) / (2.0 * H_FD)
    err = _row_scaled_error(J @ w, directional_fd)
    assert err.max() < 1e-5


@pytest.mark.parametrize("which", ["reference", "realistic"])
def test_poisson_row_couples_only_its_three_neighbours(solved_states, which):
    sol, u = solved_states[which]
    _, J = sol.residual(u, V_PROBE, want_jac=True)
    N = sol.N
    assert isinstance(J, csr_matrix)
    for i in range(1, N - 1):
        nz = J.indices[J.indptr[i]:J.indptr[i + 1]]
        assert set(int(j) for j in nz if J[i, int(j)] != 0.0) <= {
            i - 1, i, i + 1, N + i, 2 * N + i}


def test_contact_rows_are_identity():
    """The four Dirichlet rows are linear, so their Jacobian is exact by hand."""
    dev = reference_device(n_nodes=151)
    sol = DDSolver(dev)
    u = dev.initial_state(0.25)
    _, J = sol.residual(u, 0.25, want_jac=True)
    N = dev.N
    for row, col in ((0, 0), (N - 1, N - 1), (N, N), (2 * N - 1, 2 * N - 1),
                     (2 * N, 2 * N), (3 * N - 1, 3 * N - 1)):
        assert J[row, col] == 1.0
        assert np.abs(np.asarray(J[row].todense())).sum() == pytest.approx(1.0)
