"""Compatibility facade for the GC and FC numerical solvers.

New code should import the solver owned by each system directly from
``gc_solver`` or ``fc_solver``.  These exports keep the previous public API
stable for downstream users.
"""

from ._solver_common import step_count as _step_count
from .fc_solver import solve_symplectic
from .gc_solver import ExtendedSystem, solve_extended

__all__ = ["ExtendedSystem", "_step_count", "solve_extended", "solve_symplectic"]
