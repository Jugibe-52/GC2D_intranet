"""Reusable numerical formulations for structure-preserving methods."""

from contracts.formulations.projection import ProjectionFormulation
from formulations.projection import ReducedProjection, SimultaneousProjection
from formulations.state import PhysicalFormulation, DoubledFormulation
from contracts.formulations.base import (
	DirectAdjointFormulation,
	PreparedDirectAdjointFormulation,
	PreparedStageProjectedFormulation,
	StageProjectedFormulation,
)
from formulations.fc import FCSplitFormulation
from formulations.gc import (
	HarmonicCoupling,
	GCExtendedFormulation,
	GCStageProjectedFormulation,
	gc_coupling_matrix,
)

__all__ = [
	"HarmonicCoupling",
	"ProjectionFormulation", "ReducedProjection", "SimultaneousProjection",
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
