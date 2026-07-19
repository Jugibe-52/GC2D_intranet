"""Shared static contracts for configuration and simulation boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

# NumPy's shape parameter is not expressible in current stable typing. Keeping
# the dtype open here still gives callers the ndarray API instead of bare Any.
Array: TypeAlias = npt.NDArray[Any]
TrajectoryKind: TypeAlias = Literal["gc", "fc"]
InitialConditionKind: TypeAlias = Literal["random", "fixed", "selected"]
PotentialSourceKind: TypeAlias = Literal["hdf5", "mock", "hdf5_or_mock"]

# Configuration starts as untrusted JSON and may contain expanded NumPy arrays.
# ``object`` is intentional at this boundary: consumers must narrow each value.
ParameterMap: TypeAlias = dict[str, object]


class TrajectoryParams(TypedDict):
	type: TrajectoryKind
	rho: float
	eta: float


class PotentialTrajectoryParams(TrajectoryParams, total=False):
	Ntraj: int
	init: Literal["random", "fixed"]


class FourierParams(TypedDict):
	"""Normalized parameters for a Fourier potential simulation case."""

	traj_type: TrajectoryKind
	M: int
	A: float
	rho: float
	eta: float
	Ntraj: int
	Tf: int
	TimeStep: float
	ode_solver: str
	CheckEnergy: bool
	init: InitialConditionKind
	Method: NotRequired[str]
	TwoStepIntegration: NotRequired[bool]
	Tmid: NotRequired[int]
	threshold: NotRequired[float]
	thresh_b: NotRequired[float]
	x0: NotRequired[Array]
	y0: NotRequired[Array]
	modulo: NotRequired[bool]
	grid: NotRequired[bool]
	darkmode: NotRequired[bool]
	PlotResults: NotRequired[bool]
	SavePlot: NotRequired[bool]
	SaveData: NotRequired[bool]
	extension: NotRequired[str]
	dpi: NotRequired[int]
	output_dir: NotRequired[str | Path]
	output_name: NotRequired[str]


class MockPotentialParams(TypedDict, total=False):
	A: float
	M: int
	nx: int
	ny: int
	seed: int


class PotentialIntegrationParams(TypedDict, total=False):
	n_max: int


class SolverParams(TypedDict, total=False):
	"""Parameters consumed by :meth:`classes.system.System.simulate`."""

	TimeStep: float
	ode_solver: str
	CheckEnergy: bool


class OutputParams(TypedDict, total=False):
	plot: bool
	data: bool
	wrap: bool
	extension: str
	dpi: int
