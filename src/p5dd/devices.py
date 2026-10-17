"""Reference device configurations for the P5 study."""

from __future__ import annotations

import numpy as np

from .device import Device, graded_mesh
from .material import Material

# Silicon constants shared by both configurations
SI = dict(eps_r=11.7, Nc=2.8e19, Nv=1.83e19, Eg=1.12,
          mu_n0=1350.0, mu_p0=480.0, tau_n=1e-8, tau_p=1e-8, T=300.0)

LENGTH = 1.2e-2      # device length [cm] = 120 um
X_JUNCTION = 0.5 * LENGTH
ND = 1.0e17          # n side [cm^-3]
NA = 1.0e16          # p side [cm^-3]

# The equilibrium space-charge region of this 10:1 junction extends 33 nm into
# the n side, so the finest step must stay well below 10 nm for the peak field
# and both depletion edges to be resolved.
H_FINE = 4.0e-7      # [cm] = 4 nm near the junction
H_MAX = 0.5e-4       # [cm] = 50 nm in the neutral regions
SPREAD = 6.0e-4      # [cm] radius of the refined zone around the junction


def reference_device(n_nodes: int = 1601, h_fine: float = H_FINE,
                     h_max: float = H_MAX, spread: float = SPREAD,
                     **overrides) -> Device:
    """Idealised Si p-n diode used for every validation and cross-check.

    Constant mobilities and no band-gap narrowing, so the analytic Shockley /
    depletion results are exact targets and the same input can be handed to
    Sesame unchanged.
    """
    mat_kw = dict(SI)
    mat_kw.update(overrides.pop("material", {}))
    mat = Material(mobility_model="constant", bgn_model="none", **mat_kw)
    x = graded_mesh(LENGTH, X_JUNCTION, h_fine, h_max, spread, n_nodes)
    return Device(x=x, Nd=ND, Na=NA, x_j=X_JUNCTION, mat=mat, **overrides)


def realistic_device(n_nodes: int = 1601) -> Device:
    """Same geometry with doping-dependent mobility and band-gap narrowing."""
    mat = Material(mobility_model="arora", bgn_model="slotboom", **SI)
    x = graded_mesh(LENGTH, X_JUNCTION, H_FINE, H_MAX, SPREAD, n_nodes)
    return Device(x=x, Nd=ND, Na=NA, x_j=X_JUNCTION, mat=mat)


def sesame_system(device: Device, sys_path: str | None = None,
                  surface_velocity: float = 1.0e9):
    """Build the equivalent Sesame ``Builder`` system for ``device``.

    ``surface_velocity`` emulates the pinned Ohmic equilibrium densities that
    the reference solver imposes as Dirichlet rows.
    """
    import sys as _sys

    if sys_path and sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    import sesame

    m = device.mat
    material = {
        "Nc": m.Nc, "Nv": m.Nv, "Eg": m.Eg, "affinity": 4.05,
        "epsilon": m.eps_r, "mu_e": float(m.mu_n(np.full(1, device.Nd))[0]),
        "mu_h": float(m.mu_p(np.full(1, device.Na))[0]),
        "tau_e": m.tau_n, "tau_h": m.tau_p, "Et": 0.0,
    }
    sys_ = sesame.Builder(np.asarray(device.x, dtype=float), input_length="cm",
                          T=m.T)
    sys_.add_material(material)
    x_j = device.x_j
    sys_.add_donor(device.Nd, lambda pos: pos < x_j)
    sys_.add_acceptor(device.Na, lambda pos: pos >= x_j)
    sys_.contact_type("Ohmic", "Ohmic")
    sys_.contact_S(surface_velocity, surface_velocity,
                   surface_velocity, surface_velocity)
    return sys_
