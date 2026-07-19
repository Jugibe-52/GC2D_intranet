# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Result type returned by the Hamiltonian-system solvers."""

from scipy.optimize import OptimizeResult


class Solution(OptimizeResult):
    """Attribute-accessible container for a sampled numerical solution."""


# Compatibility name retained while callers migrate from the vendored solver.
OdeSolution = Solution


__all__ = ["OdeSolution", "Solution"]
