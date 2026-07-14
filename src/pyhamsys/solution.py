# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Result types returned by the solvers."""

from scipy.optimize import OptimizeResult


class OdeSolution(OptimizeResult):
    pass
