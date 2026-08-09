"""Run a short guiding-center trajectory through the canonical public API."""

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	InitialValueProblem,
	RK4,
	SimulationRequest,
	simulate,
)


potential = Potential.random(
	A=0.08,
	M=3,
	nx=16,
	ny=16,
	seed=27,
	interpolation_order=5,
)
configuration = GCInitialConfiguration.from_components(
	x=np.asarray([1.0]),
	y=np.asarray([1.2]),
)
problem = InitialValueProblem(
	GuidingCenterDynamics(potential, rho=0.05),
	configuration,
)
request = SimulationRequest.uniform(
	t_span=(0.0, 0.2),
	max_step=0.01,
	sample_count=5,
)
solution = simulate(problem, RK4(), request)

print(f"steps: {solution.diagnostics['step_count']}")
print(f"final state: {solution.states[:, -1]}")
