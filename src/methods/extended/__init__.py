"""Public ABBA and BM4 methods in the shared extended-space family."""

from methods.extended.configuration import (
	ABBA4_PROJECTION_PLACEMENTS,
	ABBA_PROJECTION_FORMULATIONS,
	ABBA_STATE_EXTENSIONS,
	ProjectionPlacement,
	ProjectionFormulation,
	StateExtension,
)
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit, ABBA2Midpoint
from methods.extended.bm4 import BM4Implicit, BM4Midpoint

from methods.extended.palindromic import PalindromicMethod, PalindromicImplicit, PalindromicMidpoint
from methods.extended.core.composition import Composition
from methods.extended.core.operations import Operation, A, B, T, C

__all__ = [
	'PalindromicMethod', 'PalindromicImplicit', 'PalindromicMidpoint', 'Composition', 'Operation', 'A', 'B', 'T', 'C',
	"BM4Implicit",
	"BM4Midpoint",
	"ABBA2Midpoint",
	"ABBA2Implicit",
	"ABBA4Implicit",
	"ABBA4_PROJECTION_PLACEMENTS",
	"ABBA_PROJECTION_FORMULATIONS",
	"ABBA_STATE_EXTENSIONS",
	"ProjectionPlacement",
	"ProjectionFormulation",
	"StateExtension",
]
