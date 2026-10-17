"""Silicon material models: mobility, band-gap narrowing, SRH recombination."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import EPS_0, einstein_mu, intrinsic_density, thermal_voltage


@dataclass
class Material:
    """Parameter set for a homogeneous material region.

    ``mobility_model`` and ``bgn_model`` select between the idealised constant
    reference model (used for the analytic / Sesame cross-checks) and the
    doping-dependent models needed for a physically realistic Si diode.
    """

    name: str = "Si"
    eps_r: float = 11.7
    Nc: float = 2.8e19          # [cm^-3]
    Nv: float = 1.83e19         # [cm^-3]
    Eg: float = 1.12            # [eV]
    mu_n0: float = 1350.0       # low-field lattice mobility [cm^2/(V s)]
    mu_p0: float = 480.0
    tau_n: float = 1e-8         # SRH lifetimes [s]
    tau_p: float = 1e-8
    mobility_model: str = "constant"   # "constant" | "arora"
    bgn_model: str = "none"            # "none" | "slotboom"
    T: float = 300.0

    # --- derived quantities -------------------------------------------------
    @property
    def Vt(self) -> float:
        return thermal_voltage(self.T)

    @property
    def eps(self) -> float:
        """Permittivity [F/cm]."""
        return self.eps_r * EPS_0

    def ni0(self) -> float:
        """Intrinsic density without band-gap narrowing [cm^-3]."""
        return intrinsic_density(self.Nc, self.Nv, self.Eg, self.T)

    def bgn(self, doping: np.ndarray | float) -> np.ndarray | float:
        """Band-gap narrowing [eV], Slotboom-type empirical form."""
        if self.bgn_model == "none":
            return np.zeros_like(np.asarray(doping, dtype=float), dtype=float)
        n = np.maximum(np.asarray(doping, dtype=float), 0.0)
        return 32.5e-3 * (n / 1e18) ** 0.125

    def ni(self, doping) -> np.ndarray | float:
        """Effective intrinsic density at a given total doping [cm^-3]."""
        base = self.ni0()
        shift = self.bgn(doping)
        return base * np.exp(shift / (2.0 * self.Vt))

    # --- mobility -----------------------------------------------------------
    def _arora(self, n_total, mu_min: float, mu_ref: float, p: float, T: float):
        """Arora-type doping dependence, scaled to 300 K values."""
        n_total = np.maximum(np.asarray(n_total, dtype=float), 1.0)
        t300 = (300.0 / T) ** 2.33
        mu = mu_min * t300 + (mu_ref - mu_min) * t300 / (1.0 + (n_total / 1.3e17) ** p)
        return mu

    def mu_n(self, n_total, field: float = 0.0):
        if self.mobility_model == "constant":
            return np.full_like(np.asarray(n_total, dtype=float), self.mu_n0, dtype=float)
        return self._arora(n_total, 92.0, 1350.0, 0.72, self.T)

    def mu_p(self, n_total, field: float = 0.0):
        if self.mobility_model == "constant":
            return np.full_like(np.asarray(n_total, dtype=float), self.mu_p0, dtype=float)
        return self._arora(n_total, 54.3, 450.0, 0.81, self.T)

    def d_n(self, n_total, field: float = 0.0):
        return einstein_mu(self.mu_n(n_total, field), self.T)

    def d_p(self, n_total, field: float = 0.0):
        return einstein_mu(self.mu_p(n_total, field), self.T)

    # --- recombination ------------------------------------------------------
    def srh(self, n, p):
        """Net SRH recombination rate R [cm^-3 s^-1] with mid-gap traps."""
        n = np.maximum(np.asarray(n, dtype=float), 0.0)
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        ni2 = self.ni(np.maximum(n + p, 1.0)) ** 2
        num = n * p - ni2
        den = self.tau_n * (p + self.ni(n + p)) + self.tau_p * (n + self.ni(n + p))
        return num / np.maximum(den, 1e-30)
