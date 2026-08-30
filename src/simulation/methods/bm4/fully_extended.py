"""Fully duplicated state-space extension of implicit BM4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .._fully_extended import (
	_FullyExtendedImplicitMethod,
	_Variant,
	_nonnegative_finite,
)


@dataclass(frozen=True, slots=True)
class BM4_implicit2(_FullyExtendedImplicitMethod):
	"""Fourth-order BM4 with full-state duplication and projection."""

	coupling_frequency: float = float(np.pi / 8.0)

	_variant: ClassVar[_Variant] = "bm4"

	def __post_init__(self) -> None:
		"""Validate shared nonlinear controls and the BM4 binding frequency."""
		_FullyExtendedImplicitMethod.__post_init__(self)
		object.__setattr__(
			self,
			"coupling_frequency",
			_nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)


__all__ = ["BM4_implicit2"]
