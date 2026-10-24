"""Compare explicit center/end time translations in a small GC simulation."""

from dataclasses import replace

import numpy as np

from dynamics.gc import GuidingCenterDynamics
from potential.potential import Potential
from simulation import (A, B, T, Composition, InitialValueProblem,
	PalindromicImplicit, PalindromicMidpoint, SimulationRequest, simulate)


def main() -> None:
	"""Keep the same field and step while changing the operation palindrome."""
	center = Composition.from_half('ABBA-center', (A(.5), B(.5), T(.5)))
	edges = Composition.from_half('ABBA-edges', (T(.5), A(.5), B(.5)))
	potential = Potential.random(A=.08, M=3, nx=16, ny=16, seed=27)
	problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=.05),
		np.array([1., 1.2]), layout=GuidingCenterDynamics.coordinates)
	request = SimulationRequest.uniform(t_span=(0., .1),  sample_count=3)
	for method in (PalindromicImplicit, PalindromicMidpoint):
		for recipe in (center, edges):
			result = simulate(problem, replace(method(recipe, track_energy=True), step=0.01), request)
			operations = ' '.join(f'{op.name}({op.coefficient:g})' for op in recipe.operations)
			print(f'{method.__name__}, {recipe.name}: {operations}')
			print(f'Final physical state: {result.states[:, -1]}')


if __name__ == '__main__':
	main()
