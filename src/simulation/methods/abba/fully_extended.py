"""Full-state implicit A-B-B-A variants built on the shared extended kernel."""

from __future__ import annotations

from typing import ClassVar

from ..bm4.fully_extended import _FullyExtendedImplicitMethod, _Variant


class ABBA_implicit2(_FullyExtendedImplicitMethod):
	"""Second-order A-B-B-A with both ``z`` and ``(t,k)`` duplicated."""

	_variant: ClassVar[_Variant] = "abba"


class ABBA4_implicit2(_FullyExtendedImplicitMethod):
	"""Fourth-order triple jump of full-state implicit A-B-B-A steps."""

	_variant: ClassVar[_Variant] = "abba4"


__all__ = ["ABBA_implicit2", "ABBA4_implicit2"]
