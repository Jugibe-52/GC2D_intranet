"""Physical stage data supplied to passive observers by numerical methods."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias

import numpy as np


class StageQuadrature(NamedTuple):
	"""Current-step evaluations and signed physical quadrature weights.

	Each weight includes its duration and any physical normalization, including
	the factor one half for doubled GC shears. States are single physical copies.
	The reduction scale preserves the method's original sum-before-normalization
	arithmetic, including its overflow behavior, without changing physical weights.
	Tuple leaves also form an ordinary JAX pytree without registering a class.
	"""

	times: tuple[Any, ...]
	states: tuple[Any, ...]
	weights: tuple[Any, ...]
	reduction_scale: Any = 1.


@dataclass(frozen=True, slots=True)
class InterpolantQuadrature:
	"""An accepted adaptive trajectory, evaluated only within its interval."""

	start_time: float
	end_time: float
	dense_state: Callable[[float | np.ndarray], np.ndarray]


QuadratureData: TypeAlias = StageQuadrature | InterpolantQuadrature

__all__ = ["StageQuadrature", "InterpolantQuadrature", "QuadratureData"]
