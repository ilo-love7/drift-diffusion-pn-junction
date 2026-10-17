"""Physical constants and unit helpers for the 1D drift-diffusion solver.

Unit system used throughout the project (practical semiconductor units):
length cm, potential V, density cm^-3, time s, current density A/cm^2.
"""

from __future__ import annotations

import numpy as np

Q_E = 1.602176634e-19      # elementary charge [C]
K_B = 1.380649e-23         # Boltzmann constant [J/K]
EPS_0 = 8.8541878128e-14   # vacuum permittivity [F/cm]


def thermal_voltage(T: float = 300.0) -> float:
    """Thermal voltage kT/q [V]."""
    return K_B * T / Q_E


def intrinsic_density(Nc: float, Nv: float, Eg: float, T: float = 300.0) -> float:
    """Intrinsic carrier density from effective densities of states [cm^-3]."""
    return float(np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * thermal_voltage(T))))


def einstein_mu(mu: float, T: float = 300.0) -> float:
    """Diffusion coefficient from mobility via Einstein relation [cm^2/s]."""
    return mu * thermal_voltage(T)
