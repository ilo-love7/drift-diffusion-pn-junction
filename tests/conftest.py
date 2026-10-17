"""Make ``src/p5dd`` and the vendored Sesame importable, plus shared solutions.

The session-scoped fixtures solve the two reference devices once; every physics
test then reuses those states instead of paying for a Newton solve again.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "reference" / "sesame"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from p5dd import DDSolver, reference_device, realistic_device  # noqa: E402

#: Node count used by the equilibrium fixtures: 801 nodes resolve the
#: space-charge region (4 nm steps) at a fraction of the cost of the 1601-node
#: production mesh, and differ from it by 1.3e-4 in the depletion width
#: (measured in case 4, which shows second-order convergence).
N_FAST = 801


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


def _solved(device) -> SimpleNamespace:
    sol = DDSolver(device)
    u, rep = sol.solve_point(0.0)
    return SimpleNamespace(dev=device, sol=sol, u=u, rep=rep,
                           st=sol.physical_state(u))


@pytest.fixture(scope="session")
def equilibrium() -> SimpleNamespace:
    """Reference device (constant mobility, no band-gap narrowing) at V = 0."""
    return _solved(reference_device(n_nodes=N_FAST))


@pytest.fixture(scope="session")
def equilibrium_bgn() -> SimpleNamespace:
    """Realistic device (Arora mobility + Slotboom band-gap narrowing) at V = 0."""
    return _solved(realistic_device(n_nodes=N_FAST))
