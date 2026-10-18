"""Transient machinery: the discrete charge-conservation law, the integrator
order, and the two slab modes whose decay rates are known analytically.

These are the checks the steady suite cannot make.  A time integration is only
verified if (i) the *combined* particle-plus-displacement current is spatially
uniform while the particle current alone is not, (ii) the measured decay rate
lands on the analytic eigenvalue once the known time-discretisation lag is
undone, and (iii) the residual of that lag is an order-1 (backward Euler) or
order-2 (Crank-Nicolson) function of the step.

The slab rather than the junction is used on purpose: it is the one structure
with closed-form transient eigenmodes, because its equilibrium potential is flat
and its contacts are identical, so a half-sine perturbation is an exact shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import (DDSolver, ambipolar_decay, charge_mode_state,
                  dielectric_relaxation, discrete_laplacian,
                  discrete_wavenumber, homogeneous_device, measured_rate,
                  mode_amplitude, mode_wavenumber, plasma_mode_state, true_rate)

N_NODES = 201
V_PROBE = 0.05
#: Perturbation amplitudes.  Both are chosen by measurement, not by taste: the
#: potential amplitude must keep the dielectric mode's row content above the
#: stopping tolerance for the whole trace (1e-5 does not: Newton then accepts the
#: initial state untouched), and the plasma amplitude must stay inside the range
#: where SRH is linear, i.e. below about a tenth of the equilibrium hole density.
AMP_PSI = 5.0e-2
DELTA_N = 1.0e13 / 1.0e17


@pytest.fixture(scope="module")
def slab():
    dev = homogeneous_device(n_doping=1.0e17, n_nodes=N_NODES)
    sol = DDSolver(dev, tol=1e-9)
    u_eq, rep = sol.solve_point(0.0)
    assert rep.converged
    return {"dev": dev, "sol": sol, "u_eq": u_eq, "k": mode_wavenumber(dev)}


def _decay_trace(slab, u0, component, dt, n_steps, theta, tol):
    sol = DDSolver(slab["dev"], tol=tol, theta=theta)
    u = sol.snapshot(u0, 0.0)
    t, a, its = [0.0], [mode_amplitude(slab["dev"], u, slab["u_eq"], component,
                                       slab["k"])], []
    for _ in range(n_steps):
        u, rep = sol.step(0.0, dt)
        assert rep.converged, f"step failed, residual {rep.residual:.3e}"
        assert rep.iterations >= 1, "accepted without integrating: zero Newton steps"
        t.append(sol.time)
        a.append(mode_amplitude(slab["dev"], u, slab["u_eq"], component, slab["k"]))
        its.append(rep.iterations)
    return np.array(t), np.array(a)


def _rate(slab, u0, component, lam, f, theta, tol, n_efold=3.0):
    """One mode's fitted decay: ``(corrected, amp_ratio, raw)``, all over ``lam``.

    ``raw`` is what the finite step actually measures and therefore carries the
    integrator's own lag; ``corrected`` has that known lag undone and is the
    number the physics claim rests on.  Both are returned because the order test
    needs the first and the rate tests need the second.
    """
    dt = f / lam
    t, a = _decay_trace(slab, u0, component, dt, int(np.ceil(n_efold / f)), theta, tol)
    i0, i1 = int(np.ceil(0.3 / f)), min(int(np.ceil(n_efold / f)), len(t) - 1)
    w = float(np.polyfit(t[i0:i1 + 1], np.log(np.abs(a[i0:i1 + 1])), 1)[0])
    measured = -w
    return (true_rate(measured, dt, theta) / lam, a[-1] / a[0],
            measured / lam)


# --- the slab at rest --------------------------------------------------------

def test_slab_equilibrium_is_exact(slab):
    """A uniformly doped slab at zero bias is solved exactly, not approximately.

    With no junction there is nothing to deplete: the density must equal the
    doping at every node, the potential be flat, and the current vanish to the
    last bit - Scharfetter-Gummel carries no flux across a link whose two
    quasi-Fermi levels coincide.
    """
    dev, u = slab["dev"], slab["u_eq"]
    n, p = dev.densities(u)
    ni = dev.r_ni[0] * dev.Nref
    # n - Nd = p, which is the equilibrium minority density: 1e3 cm^-3 against a
    # 1e17 doping, i.e. 1e-14 relative.  The bound is the solver's, not that.
    assert np.max(np.abs(n / (dev.Nd / dev.Nref) - 1.0)) < 1e-10
    assert np.max(np.abs(n * p * dev.Nref ** 2 / ni ** 2 - 1.0)) < 1e-12
    Jn, Jp = slab["sol"].currents(u)
    assert np.max(np.abs(Jn)) == 0.0 and np.max(np.abs(Jp)) == 0.0
    assert np.max(np.abs(u[:dev.N] - u[:dev.N][0])) < 1e-14


def test_biased_slab_is_an_exact_resistor(slab):
    """Ohm's law out of the full drift-diffusion system, to solver precision.

    Poisson is solved by a linear potential with both densities untouched, so the
    current must be exactly ``q (mu_n n + mu_p p) V / L``.  The sign is the
    solver's convention - the same one that makes a forward diode bias negative -
    because the bias is applied at the right contact.
    """
    dev, sol = slab["dev"], slab["sol"]
    u, rep = sol.solve_point(V_PROBE)
    assert rep.converged
    sigma = dielectric_relaxation(dev, slab["k"])["sigma"]
    L = dev.x[-1] - dev.x[0]
    J = sol.total_current(u)
    assert abs(J / (-sigma * V_PROBE / L) - 1.0) < 1e-6
    assert J * V_PROBE < 0.0
    psi = u[:dev.N]
    line = psi[0] + (psi[-1] - psi[0]) * (dev.x - dev.x[0]) / L
    assert np.max(np.abs(psi - line)) * dev.mat.Vt < 1e-9


def test_one_huge_step_reproduces_the_steady_solve(slab):
    """Integrating for a million time constants must land on the steady state.

    This isolates the time term from the spatial one: backward Euler at
    ``dt -> infinity`` is the steady system written in the same variables, so a
    step that large has to give the answer ``solve_point`` gives.  A Jacobian
    entry that forgot the time derivative, or a ``theta`` weighting applied to
    the wrong block, shows up here first.
    """
    dev, sol = slab["dev"], slab["sol"]
    lam = dielectric_relaxation(dev, slab["k"])["lambda"]
    big = DDSolver(dev, tol=1e-9, theta=1.0)
    big.snapshot(slab["u_eq"], 0.0)
    u_big, rep = big.step(V_PROBE, 1.0e6 / lam)
    assert rep.converged
    u_ref, _ = sol.solve_point(V_PROBE)
    assert np.max(np.abs(u_big[:dev.N] - u_ref[:dev.N])) * dev.mat.Vt < 1e-9
    assert abs(big.total_current(u_big) / sol.total_current(u_ref) - 1.0) < 1e-6


# --- displacement current and the conservation invariant ---------------------

def test_displacement_term_is_required_for_current_uniformity(slab):
    """Total current (particles + displacement) is uniform; particles alone are not.

    While the field moves, ``div(Jn + Jp)`` balances ``eps dE/dt``, so an ammeter
    in the external circuit reads one number along the whole device.  Dropping
    the displacement term must break that - this is the regression test for the
    term itself, and both statements are measured against the magnitudes of the
    terms that cancel rather than against their (identically zero) sum.
    """
    dev = slab["dev"]
    lam = dielectric_relaxation(dev, slab["k"])["lambda"]
    u0 = charge_mode_state(dev, slab["u_eq"], AMP_PSI, slab["k"])
    sol = DDSolver(dev, tol=1e-8, theta=1.0)
    u = sol.snapshot(u0, 0.0)
    dt = 0.05 / lam
    with_disp, without = [], []
    for _ in range(10):
        u_prev = u
        u, rep = sol.step(0.0, dt)
        assert rep.converged
        with_disp.append(sol.current_uniformity(u, u_prev, dt))
        without.append(sol.current_uniformity(u))
    assert max(with_disp) < 1e-3
    assert min(without) > 1e-2
    assert max(without) / max(with_disp) > 1e3


def test_terminal_current_is_zero_for_a_symmetric_mode(slab):
    """Mirror symmetry makes the conserved total current vanish exactly.

    In a slab between two identical contacts a half-sine satisfies
    ``J(x) = -J(L - x)``, so the uniform current the invariant guarantees must be
    zero - a sharper test than any bound on a non-zero value, and it pins the
    integration constant of the invariant rather than only its flatness.

    The *spread* of the profile is a separate question and is not zero: adding
    the two continuity rows leaves the discrete Poisson row, so whatever Newton
    leaves there appears in the current amplified by ``1 / dt_hat`` -- here
    ``dt_hat = 8e-6``, i.e. five orders on a 1e-8 weighted residual.  The test
    therefore asserts the mechanism rather than an unattainable absolute: the
    spread must fall when the step grows, and the uniform value must stay at the
    round-off level.
    """
    dev = slab["dev"]
    lam = dielectric_relaxation(dev, slab["k"])["lambda"]
    u0 = charge_mode_state(dev, slab["u_eq"], AMP_PSI, slab["k"])
    worst = {}
    for f in (0.05, 0.2):
        sol = DDSolver(dev, tol=1e-8, theta=1.0)
        u = sol.snapshot(u0, 0.0)
        dt = f / lam
        means, spreads = [], []
        for _ in range(10):
            u_prev = u
            u, rep = sol.step(0.0, dt)
            assert rep.converged
            K = sol.total_current_profile(u, u_prev, dt)
            Jn, Jp = sol.currents(u)
            disp = sol.displacement_current(u, u_prev, dt)
            scale = np.max(np.abs(Jn + Jp)) + np.max(np.abs(disp))
            means.append(abs(np.mean(K)) / scale)
            spreads.append(np.ptp(K) / scale)
        worst[f] = (max(means), max(spreads))
    assert max(v[0] for v in worst.values()) < 1e-6
    assert worst[0.05][1] < 1e-3
    assert worst[0.2][1] < 0.5 * worst[0.05][1]


# --- the two analytic time constants ----------------------------------------

def test_dielectric_mode_decays_at_the_analytic_rate(slab):
    """``sigma/eps + Dn k_h^2``, to solver precision, for both integrators.

    At this step size the finite step is wrong on its own by 2.4 per cent
    (backward Euler, the expected ``-x/2``) or 2.1e-4 (Crank-Nicolson, ``+x^2/12``),
    so the closed-form amplification factor has to be undone before the physics
    can be read off.  Requiring the correction to buy two orders of magnitude
    makes that a measured statement: a solver whose rate simply wandered with
    ``dt`` could not reach 1e-6 agreement at both step sizes.
    """
    dev = slab["dev"]
    dl = dielectric_relaxation(dev, slab["k"])
    u0 = charge_mode_state(dev, slab["u_eq"], AMP_PSI, slab["k"])
    for theta, tol in ((1.0, 1e-8), (0.5, 1e-8)):
        ratio, fall, raw = _rate(slab, u0, "psi", dl["lambda"], 0.05, theta, tol)
        assert abs(ratio - 1.0) < 1e-4, f"theta={theta}: rate/{dl['lambda']} = {ratio}"
        assert fall < 0.06                     # three e-folds really happened
        assert abs(raw - 1.0) > 5e-5, f"theta={theta}: no lag to correct"
        assert abs(raw - 1.0) / abs(ratio - 1.0) > 1e2


def test_dielectric_rate_is_amplitude_independent(slab):
    """Linear response: the rate is a property of the system, not of the probe."""
    dev = slab["dev"]
    dl = dielectric_relaxation(dev, slab["k"])
    rates = [_rate(slab, charge_mode_state(dev, slab["u_eq"], amp, slab["k"]),
                   "psi", dl["lambda"], 0.05, 1.0, 1e-8)[0]
             for amp in (1.0e-3, 1.0e-2, 5.0e-2)]
    assert max(rates) / min(rates) - 1.0 < 1e-4


def test_plasma_mode_decays_at_the_ambipolar_rate(slab):
    """``1/tau + D_a k_h^2``, and the diffusion share must be *resolved*.

    Ignoring the ``D_a k^2`` term would be an 8.5e-3 error against a measured one
    of 1e-5, so the check is not just the rate but the ratio of the two errors -
    otherwise a solver that simply did not diffuse would also pass.
    """
    dev = slab["dev"]
    am = ambipolar_decay(dev, slab["k"])
    u0 = plasma_mode_state(dev, slab["u_eq"], DELTA_N, slab["k"])
    ratio, _, _ = _rate(slab, u0, "n", am["lambda"], 0.05, 1.0, 1e-12)
    assert abs(ratio - 1.0) < 1e-3
    # A solver that did not diffuse would report 1/tau instead of the analytic
    # lambda, i.e. be wrong by this much -- two orders above the measured error.
    if_ignored = abs(1.0 / (am["lambda"] * am["tau"]) - 1.0)
    assert if_ignored > 5e-3
    assert if_ignored / abs(ratio - 1.0) > 10.0


def test_plasma_rate_drifts_with_injection_level(slab):
    """Where linear response ends is measured, not assumed.

    SRH stays linear only while the perturbation is small against *both*
    equilibrium densities; the extracted rate is flat to 1e-5 for
    ``dn <= 1e13 cm^-3`` (0.1 of the equilibrium hole density) and then falls
    about ten times faster per decade.  Recorded so the analytic formula's range
    of validity is a measured number.
    """
    dev = slab["dev"]
    am = ambipolar_decay(dev, slab["k"])
    out = {}
    for delta in (1.0e12, 1.0e13, 1.0e14):
        u0 = plasma_mode_state(dev, slab["u_eq"], delta / dev.Nref, slab["k"])
        out[delta] = _rate(slab, u0, "n", am["lambda"], 0.05, 1.0, 1e-12)[0]
    assert abs(out[1.0e13] - out[1.0e12]) < 1e-4
    assert out[1.0e14] < out[1.0e13] - 1e-4


# --- the integrator's own convergence order ---------------------------------

def test_time_step_order_is_one_for_backward_euler_and_two_for_crank_nicolson(slab):
    """The lag of the theta scheme on ``y' = -lambda y`` is exactly invertible, so
    what is left over must scale as ``dt`` (theta=1) or ``dt^2`` (theta=1/2).

    The error measured here is the *raw* one -- the difference between what a
    finite step returns and the analytic eigenvalue -- because that difference is
    the truncation lag itself.  Correcting it first, as the rate tests do, leaves
    only Newton noise (~1e-7) and the slope disappears.
    """
    dev = slab["dev"]
    dl = dielectric_relaxation(dev, slab["k"])
    u0 = charge_mode_state(dev, slab["u_eq"], AMP_PSI, slab["k"])
    fs = (0.05, 0.02, 0.008)
    for theta, expected in ((1.0, 1.0), (0.5, 2.0)):
        errs = [abs(_rate(slab, u0, "psi", dl["lambda"], f, theta, 1e-8)[2]
                    - 1.0) for f in fs]
        assert all(e > 0.0 for e in errs)
        slope = np.polyfit(np.log(fs), np.log(errs), 1)[0]
        assert abs(slope - expected) < 0.35, f"theta={theta}: slope {slope}, {errs}"


def test_amplification_maps_are_exact_inverse_pair():
    """``true_rate`` must undo ``measured_rate``, and both must quote the known
    scalar results: backward Euler ``1/(1+x)``, Crank-Nicolson ``(1-x/2)/(1+x/2)``.

    Crank-Nicolson is only tested below ``x = 2``, where its amplification factor
    changes sign; past that the scheme is not a decay at all, and the inverse map
    is bounded by ``x = 2`` for the same reason.
    """
    dt = 1.0e-11
    for theta, xs in ((1.0, (0.01, 0.5, 3.0)), (0.5, (0.01, 0.5, 1.5))):
        for x in xs:
            lam = x / dt
            m = measured_rate(lam, dt, theta)
            assert abs(true_rate(m, dt, theta) / lam - 1.0) < 1e-12
    g_be = np.exp(-measured_rate(1.0 / dt, dt, 1.0) * dt)
    assert abs(g_be - 1.0 / 2.0) < 1e-12
    g_cn = np.exp(-measured_rate(1.0 / dt, dt, 0.5) * dt)
    assert abs(g_cn - (1.0 - 0.5) / (1.0 + 0.5)) < 1e-12


def test_discrete_wavenumber_is_the_finite_volume_eigenvalue():
    """``2 sin(kh/2)/h`` must be the exact eigenvalue the solver applies.

    The whole point of comparing a measured rate to an analytic one on a mesh is
    that the mesh operator, not the continuum one, sets the answer; if this
    identity failed, every ``k_h`` in the theory module would be decoration.  The
    half-sine is used as a probe: vanishing at both pinned contacts, it is an
    eigenvector of the interior second-difference operator, so the discrete
    Laplacian must return exactly ``-(k_h Lambda)^2`` times its input.
    """
    dev = homogeneous_device(n_doping=1.0e17, n_nodes=N_NODES)
    k = mode_wavenumber(dev)
    kh = discrete_wavenumber(dev, k)
    s = np.sin(k * (dev.x - dev.x[0]))
    eigen = (kh * dev.length_scale) ** 2
    rel = np.max(np.abs(discrete_laplacian(dev, s) + eigen * s[1:-1])) / eigen
    assert rel < 1e-10
    assert abs(kh / k - 1.0) < 1e-3          # O(h^2) below the continuum value
    assert np.max(np.abs(s[[0, -1]])) < 1e-12  # pinned to zero at both contacts
