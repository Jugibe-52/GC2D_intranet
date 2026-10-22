"""Structural contracts consumed by visualization.trajectory_symplecticity."""

from __future__ import annotations

from typing import Protocol


class TrajectorySymplecticityRecordView(Protocol):
	"""Scalar record fields consumed by the comparison plot."""

	time: float
	mean_local_relative_defect: float
	std_local_relative_defect: float
	mean_accumulated_relative_defect: float
	std_accumulated_relative_defect: float


__all__ = ['TrajectorySymplecticityRecordView']
