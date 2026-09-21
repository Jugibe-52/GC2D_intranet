"""Reusable numerical formulations for structure-preserving methods."""

from .base import (
	DirectAdjointFormulation,
	PreparedDirectAdjointFormulation,
	PreparedStageProjectedFormulation,
	StageProjectedFormulation,
)
from .fc import FCSplitFormulation
from .gc import (
	GCExtendedFormulation,
	GCStageProjectedFormulation,
	gc_coupling_matrix,
)

__all__ = [
	"DirectAdjointFormulation",
	"FCSplitFormulation",
	"GCExtendedFormulation",
	"GCStageProjectedFormulation",
	"gc_coupling_matrix",
	"PreparedDirectAdjointFormulation",
	"PreparedStageProjectedFormulation",
	"StageProjectedFormulation",
]
