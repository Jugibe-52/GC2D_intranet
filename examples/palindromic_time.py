"""Compare explicit center/end time translations in a small GC simulation."""

from diagnostics.methods.energy import EnergyObserver

from dataclasses import replace

import numpy as np

from dynamics.gc import GuidingCenterDynamics
from potential.potential import Potential
from simulation import (A, B, T, Composition, InitialValueProblem,
	PalindromicImplicit, PalindromicMidpoint, SimulationRequest, simulate)


def main() -> None:
	"""Keep the same field and step while changing the operation palindrome."""
	center: Composition = (A(.5), B(.5), T(1.), B(.5), A(.5))
	edges: Composition = (T(.5), A(.5), B(.5), B(.5), A(.5), T(.5))
	potential = Potential.random(A=.08, M=3, nx=16, ny=16, seed=27)
	problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=.05),
		np.array([1., 1.2]), layout=GuidingCenterDynamics.coordinates)
	request = SimulationRequest.uniform(t_span=(0., .1),  sample_count=3)
	for method in (PalindromicImplicit, PalindromicMidpoint):
		for name, recipe in (('ABBA-center', center), ('ABBA-edges', edges)):
			result = simulate(problem.with_request(request), replace(method(recipe, composition_name=name), step=0.01), observers=(EnergyObserver(),))
			operations = ' '.join(f'{op.name}({op.coefficient:g})' for op in recipe)
			print(f'{method.__name__}, {name}: {operations}')
			print(f'Final physical state: {result.states[-1]}')


if __name__ == '__main__':
	main()
