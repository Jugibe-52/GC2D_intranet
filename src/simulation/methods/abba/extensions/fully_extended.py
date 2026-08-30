"""Fully duplicated state-space extensions of implicit ABBA methods."""

from __future__ import annotations

from typing import ClassVar

from ..._fully_extended import _FullyExtendedImplicitMethod, _Variant


class ABBA2FullyExtendedImplicit(_FullyExtendedImplicitMethod):
	"""Second-order ABBA with both ``z`` and ``(t,k)`` duplicated."""

	_variant: ClassVar[_Variant] = "abba"


class ABBA4FullyExtendedImplicit(_FullyExtendedImplicitMethod):
	"""Fourth-order triple jump of full-state implicit A-B-B-A steps."""

	_variant: ClassVar[_Variant] = "abba4"


__all__ = ["ABBA2FullyExtendedImplicit", "ABBA4FullyExtendedImplicit"]
