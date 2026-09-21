"""Public order-2 implicit ABBA configuration on the shared family runtime."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Literal
from ._implicit import _ABBAImplicitMethod


@dataclass(slots=True)
class ABBA2Implicit(_ABBAImplicitMethod):
	"""Second-order implicit ABBA with optional physical energy tracking.

	The two projection formulations define the same accepted map and may be
	solved by Newton or Broyden. Only the spatial state is duplicated and projected. Physical energy tracking is a
	triangular auxiliary update and does not change the accepted physical map.
	"""

	order: ClassVar[Literal[2, 4, 6]] = 2


__all__ = ["ABBA2Implicit"]
