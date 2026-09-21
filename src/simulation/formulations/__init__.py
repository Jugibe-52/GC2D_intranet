"""Reusable numerical formulations for structure-preserving methods."""

from .state import PhysicalFormulation, DoubledFormulation
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
	"PhysicalFormulation", "DoubledFormulation",
	"DirectAdjointFormulation",
	"FCSplitFormulation",
	"GCExtendedFormulation",
	"GCStageProjectedFormulation",
	"gc_coupling_matrix",
	"PreparedDirectAdjointFormulation",
	"PreparedStageProjectedFormulation",
	"StageProjectedFormulation",
]
