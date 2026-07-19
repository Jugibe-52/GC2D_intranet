"""Trajectory models independent from potentials and solvers."""

from typing import Any

from .fc import TrajectoryFC
from .gc import TrajectoryGC
from .trajectory import InitializationKind, Trajectory


def create_trajectory(kind: str, **params: Any) -> Trajectory:
	"""Create a GC or FC trajectory; ``fo`` is accepted as a legacy alias."""
	if kind == "gc":
		return TrajectoryGC(**params)
	if kind in {"fc", "fo"}:
		return TrajectoryFC(**params)
	raise ValueError(f"Unsupported trajectory kind: {kind!r}.")


__all__ = [
	"InitializationKind",
	"Trajectory",
	"TrajectoryFC",
	"TrajectoryGC",
	"create_trajectory",
]
