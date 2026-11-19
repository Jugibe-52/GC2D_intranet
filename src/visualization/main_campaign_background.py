"""Sample the saved campaign's Hamiltonian and GC drift without integration."""
from __future__ import annotations

from typing import Any
import numpy as np

from dynamics.gc import GuidingCenterDynamics
from potential.load import GC2DH5Metadata
from potential.potential import Potential


def validate_background_source(field: Potential, metadata: dict[str, Any]) -> None:
	"""Reject reconstructed physics that differs from a direct-radius archive."""
	provenance: Any = field.metadata
	if (not isinstance(provenance, GC2DH5Metadata)
			or not provenance.spatial_rescaling or not provenance.temporal_rescaling
			or field.spatial_extension != 'reflected' or field.interpolation_order != 3
			or field.grid.x0 != 0. or field.grid.y0 != 0. or field.grid.period != 1.
			or field.spatial_period != 2. or not np.array_equal(provenance.source_field_indices, [15])
			or metadata.get('schema_version') != 2 or metadata.get('rho_runtime') != metadata['rho']
			or metadata.get('spatial_extension') != 'reflected' or metadata.get('spatial_period') != 2.
			or metadata.get('source_field_indices') != [15] or metadata.get('sigma') is not None
			or metadata.get('magnetic_field') != 1.5
			or metadata.get('energy_evaluation') != 'physical_hamiltonian_at_saved_states'):
		raise ValueError('Backgrounds require the recorded direct-radius, reflected campaign physics.')
	if (metadata['spatial_scale_m'] != provenance.spatial_scale
			or metadata['time_scale_seconds'] != provenance.time_scale
			or metadata['rho_m'] != metadata['rho'] * provenance.spatial_scale
			or not np.isclose(metadata['cycle_duration'], 1 / field.frequencies[0], rtol=1e-13, atol=0.)
			or not np.isclose(metadata['cycle_duration'], 1., rtol=1e-13, atol=0.)):
		raise ValueError('Background radius, space or clock differs from the saved campaign.')


def sample_campaign_background(field: Potential, metadata: dict[str, Any], *,
		grid_size: int = 128, vector_grid_size: int = 33) -> dict[str, Any]:
	"""Evaluate H and f over the true period-two cell at 51 independent phases.

	Arrays are (phase, y, x), with a final xy axis for velocities. The loader
	already supplies base-cell/cycle coordinates: H and f need no extra factor.
	Evaluating the reflected potential directly preserves arrow orientation in
	all four tiles; no coordinate folding or electrostatic-field substitution occurs.
	"""
	validate_background_source(field, metadata)
	for size in (grid_size, vector_grid_size):
		if type(size) is not int or size < 2:
			raise ValueError('Background grid sizes must be integers of at least two.')
	dynamics: GuidingCenterDynamics = GuidingCenterDynamics(field, rho=metadata['rho'], B=metadata['magnetic_field'])
	phases: np.ndarray = np.linspace(0., 1., 51)
	q: np.ndarray = 2 * (np.arange(grid_size) + .5) / grid_size
	vq: np.ndarray = 2 * (np.arange(vector_grid_size) + .5) / vector_grid_size
	xx: np.ndarray
	yy: np.ndarray
	vx: np.ndarray
	vy: np.ndarray
	xx, yy = np.meshgrid(q, q)
	vx, vy = np.meshgrid(vq, vq)
	state: np.ndarray = np.r_[vx.ravel(), vy.ravel()]
	times: np.ndarray = phases * metadata['cycle_duration']
	potential: np.ndarray = np.stack([dynamics.hamiltonian_potential.evaluate(t, xx, yy) for t in times])
	velocity: np.ndarray = np.stack([dynamics.vector_field(t, state).reshape(2, vector_grid_size,
		vector_grid_size).transpose(1, 2, 0) for t in times])
	endpoint_errors: dict[str, float] = {}
	for name, values in (('potential', potential), ('velocity', velocity)):
		if not np.isfinite(values).all():
			raise ValueError('Background fields must be finite at every phase.')
		scale: float = max(float(np.abs(values).max()), 1e-30)
		np.testing.assert_allclose(values[-1], values[0], rtol=1e-11, atol=1e-13 * scale)
		endpoint_errors[name] = float(np.abs(values[-1] - values[0]).max())
		values[-1] = values[0]  # Exact display closure only after independent endpoint evaluation.
	return dict(potential=potential, velocity=velocity, phases=phases.tolist(),
		bounds=dict(x=0., y=0., span=2.), minimum=float(potential.min()), maximum=float(potential.max()),
		maxSpeed=float(np.linalg.norm(velocity, axis=-1).max()),
		units='cellule² / cycle', velocityUnits='cellule / cycle',
		phaseEndpointsVerified=True, endpointMaximumErrors=endpoint_errors,
		trajectoryIntegrations=0, rho=metadata['rho'], rhoRuntime=dynamics.rho, rhoM=metadata['rho_m'],
		source_sha256=metadata['source_sha256'], code_sha256=metadata['code_sha256'])


__all__ = ['sample_campaign_background', 'validate_background_source']
