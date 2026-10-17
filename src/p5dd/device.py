"""Device definition: graded mesh, abrupt junction, nondimensional scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import Q_E
from .material import Material


def graded_mesh(L: float, x_j: float, h_fine: float, h_max: float,
                spread: float, n_nodes: int) -> np.ndarray:
    """Smoothly graded 1D mesh with ``n_nodes`` sites on [0, L] [cm].

    The target spacing is ``h_fine`` at the junction and grows towards
    ``h_max`` once ``|x - x_j|`` exceeds ``spread``, following a squared
    profile.  Nodes are placed by equal arc length in index space, which keeps
    the spacing ratio between neighbours smooth (no abrupt size jumps).
    """
    dense = np.linspace(0.0, L, 20001)
    d = np.abs(dense - x_j) / spread
    h_of_x = h_fine + (h_max - h_fine) * np.minimum(d, 1.0) ** 2
    inv_h = 1.0 / h_of_x
    seg = dense[1:] - dense[:-1]
    index_space = np.concatenate(([0.0], np.cumsum(0.5 * (inv_h[1:] + inv_h[:-1]) * seg)))
    target = np.linspace(0.0, index_space[-1], n_nodes)
    x = np.interp(target, index_space, dense)
    x[0], x[-1] = 0.0, L
    return np.unique(x)


@dataclass
class Device:
    """1D abrupt p-n junction between two Ohmic contacts."""

    x: np.ndarray                 # site coordinates [cm], ascending
    Nd: float                     # donor density of the n side [cm^-3]
    Na: float                     # acceptor density of the p side [cm^-3]
    x_j: float                    # junction position [cm]
    mat: Material
    length_scale: float = 1.0e-4  # Lambda [cm], 1 um

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        if x.ndim != 1 or x.size < 5 or np.any(np.diff(x) <= 0):
            raise ValueError("mesh must be a strictly ascending 1D array")
        self.x = x
        self.N = x.size
        self.h = np.diff(x)                          # link lengths [cm]
        hb = np.empty(self.N)
        hb[0] = self.h[0] / 2.0
        hb[-1] = self.h[-1] / 2.0
        hb[1:-1] = (self.h[:-1] + self.h[1:]) / 2.0
        self.hb = hb / self.length_scale             # dimensionless
        self.hd = self.h / self.length_scale
        self.xd = x / self.length_scale

        Vt = self.mat.Vt
        self.Nref = max(self.Nd, self.Na, self.mat.ni0())
        self.Dref = max(self.mat.mu_n0, self.mat.mu_p0) * Vt
        self.Jscale = Q_E * self.Nref * self.Dref / self.length_scale    # A/cm^2
        self.Rscale = self.Nref * self.Dref / self.length_scale ** 2     # cm^-3 s^-1
        self.alpha = Q_E * self.Nref * self.length_scale ** 2 / (self.mat.eps * Vt)

        in_n = x < self.x_j
        self.nd = np.where(in_n, self.Nd, 0.0) / self.Nref
        self.na = np.where(in_n, 0.0, self.Na) / self.Nref
        self.cdot = self.na - self.nd                        # (NA - ND)/Nref
        self.doping_tot = (self.nd + self.na) * self.Nref     # cm^-3, model input

        self.r_ni = self.mat.ni(self.doping_tot) / self.Nref   # ni/Nref per site
        self.dn_ref = self.mat.mu_n(self.doping_tot) * Vt / self.Dref
        self.dp_ref = self.mat.mu_p(self.doping_tot) * Vt / self.Dref
        self.dn_link = 0.5 * (self.dn_ref[:-1] + self.dn_ref[1:])
        self.dp_link = 0.5 * (self.dp_ref[:-1] + self.dp_ref[1:])

        # Dirichlet values of the electrostatic potential at the contacts,
        # dimensionless: chosen so that each contact holds its equilibrium
        # majority density while its quasi-Fermi level sits at the voltage
        # reference.
        self.psi_left = float(np.log(self.nd[0] / self.r_ni[0]))
        self.psi_right0 = -float(np.log(self.na[-1] / self.r_ni[-1]))

    # ---------------------------------------------------------------- helpers
    @property
    def Vbi(self) -> float:
        """Built-in potential [V] implied by the contact Dirichlet values."""
        return (self.psi_left - self.psi_right0) * self.mat.Vt

    def densities(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slotboom transform returning dimensionless (n, p)."""
        N = self.N
        psi, phin, phip = u[:N], u[N:2 * N], u[2 * N:]
        n = self.r_ni * np.exp(np.clip(psi - phin, -60.0, 60.0))
        p = self.r_ni * np.exp(np.clip(phip - psi, -60.0, 60.0))
        return n, p

    def depletion_width(self, V: float) -> tuple[float, float, float]:
        """Analytic abrupt-junction (x_n, x_p, W) [cm] at bias ``V``."""
        vj = max(self.Vbi - V, 1e-6)
        W = np.sqrt(2.0 * self.mat.eps * vj / Q_E * (1.0 / self.Na + 1.0 / self.Nd))
        x_n = W * self.Na / (self.Na + self.Nd)
        return x_n, W - x_n, W

    def initial_state(self, V: float = 0.0) -> np.ndarray:
        """Depletion-approximation initial guess for bias ``V`` [V]."""
        Vt = self.mat.Vt
        x_n, x_p, _ = self.depletion_width(V)
        lam = self.length_scale
        x_n, x_p = x_n / lam, x_p / lam
        a_n = self.alpha * self.nd[0]
        a_p = self.alpha * self.na[-1]
        pl = self.psi_left
        pr = self.psi_right0 + V / Vt
        d = self.xd
        x_j, lo, hi = self.x_j / lam, self.x_j / lam - x_n, self.x_j / lam + x_p
        psi = np.where(
            d < lo, pl,
            np.where(
                d <= x_j, pl - 0.5 * a_n * (d - lo) ** 2,
                np.where(
                    d <= hi,
                    pl - 0.5 * a_n * x_n ** 2 - a_n * x_n * (d - x_j)
                    + 0.5 * a_p * (d - x_j) ** 2,
                    pr,
                ),
            ),
        )
        u = np.zeros(3 * self.N)
        u[:self.N] = psi
        return u
