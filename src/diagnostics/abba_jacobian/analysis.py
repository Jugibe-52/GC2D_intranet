"""Spectral and singular-value analysis of planar step Jacobians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


SpectralClass: TypeAlias = Literal["hyperbolic", "elliptic", "parabolic"]
SPECTRAL_CLASSES: tuple[SpectralClass, ...] = (
	"hyperbolic",
	"elliptic",
	"parabolic",
)


def line_angle(vector: np.ndarray) -> float:
	"""Return an unoriented planar line angle in ``[-pi/2, pi/2)``."""
	value = np.asarray(vector, dtype=float)
	if value.shape != (2,) or not np.all(np.isfinite(value)):
		raise ValueError("A line direction must be a finite planar vector.")
	norm = float(np.linalg.norm(value))
	if norm <= np.finfo(float).tiny:
		raise ValueError("A line direction must be non-zero.")
	angle = float(np.arctan2(value[1], value[0]))
	return float((angle + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _canonical_complex_vector(vector: np.ndarray) -> np.ndarray:
	"""Normalize one eigenvector and remove its arbitrary complex phase."""
	value = np.asarray(vector, dtype=complex)
	norm = float(np.linalg.norm(value))
	if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
		raise ValueError("An eigenvector must have a finite non-zero norm.")
	value = value / norm
	pivot = int(np.argmax(np.abs(value)))
	value = value * np.exp(-1j * np.angle(value[pivot]))
	# Removing insignificant imaginary residue makes real eigendirections stable.
	threshold = 64.0 * np.finfo(float).eps
	if float(np.max(np.abs(value.imag))) <= threshold:
		value = value.real.astype(complex)
	return np.asarray(value, dtype=complex)


def _ordered_eigensystem(
	matrix: np.ndarray,
	spectral_class: SpectralClass,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return deterministically ordered eigenvalues and column eigenvectors."""
	eigenvalues, eigenvectors = np.linalg.eig(matrix)
	eigenvalues = np.asarray(eigenvalues, dtype=complex)
	eigenvectors = np.asarray(eigenvectors, dtype=complex)
	if spectral_class == "hyperbolic":
		order = np.argsort(np.abs(eigenvalues), kind="stable")
	elif spectral_class == "elliptic":
		# The positive-imaginary member represents the reported rotation branch.
		order = np.argsort(-eigenvalues.imag, kind="stable")
	else:
		order = np.lexsort((eigenvalues.imag, eigenvalues.real))
	eigenvalues = eigenvalues[order]
	eigenvectors = eigenvectors[:, order]
	for column in range(2):
		eigenvectors[:, column] = _canonical_complex_vector(
			eigenvectors[:, column]
		)
	return eigenvalues, eigenvectors


def _canonical_right_singular_vectors(vectors: np.ndarray) -> np.ndarray:
	"""Fix the arbitrary sign of each real right singular vector."""
	result = np.asarray(vectors, dtype=float).copy()
	for column in range(2):
		pivot = int(np.argmax(np.abs(result[:, column])))
		if result[pivot, column] < 0.0:
			result[:, column] *= -1.0
	return result


@dataclass(frozen=True, slots=True)
class ParticleJacobianAnalysis:
	"""Intrinsic matrix, spectral, and SVD data for one planar particle."""

	jacobian: np.ndarray
	trace: float
	determinant: float
	discriminant: float
	discriminant_tolerance: float
	spectral_class: SpectralClass
	condition_number: float
	spectral_radius: float
	eigenvalue_separation: float
	eigenvector_condition_number: float
	eigendirections_defined: bool
	eigenvalues: np.ndarray
	eigenvectors: np.ndarray
	eigenvector_line_angles: np.ndarray
	singular_values: np.ndarray
	right_singular_vectors: np.ndarray
	singular_directions_defined: bool
	singular_vector_line_angles: np.ndarray


def analyze_particle_jacobian(
	jacobian: np.ndarray,
	*,
	discriminant_relative_tolerance: float = 1e-10,
) -> ParticleJacobianAnalysis:
	"""Classify and decompose one finite real ``2 x 2`` Jacobian.

	The discriminant tolerance is relative to the characteristic-polynomial
	terms. Near a repeated eigenvalue the result is deliberately classified as
	parabolic because eigendirections are not numerically reliable there.
	"""
	matrix = np.asarray(jacobian, dtype=float)
	if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
		raise ValueError("A particle Jacobian must be a finite real 2 x 2 matrix.")
	tolerance = float(discriminant_relative_tolerance)
	if not np.isfinite(tolerance) or tolerance <= 0.0:
		raise ValueError(
			"`discriminant_relative_tolerance` must be positive and finite."
		)

	trace = float(np.trace(matrix))
	determinant = float(np.linalg.det(matrix))
	discriminant = trace**2 - 4.0 * determinant
	discriminant_scale = max(1.0, trace**2, 4.0 * abs(determinant))
	discriminant_tolerance = tolerance * discriminant_scale
	if discriminant > discriminant_tolerance:
		spectral_class: SpectralClass = "hyperbolic"
	elif discriminant < -discriminant_tolerance:
		spectral_class = "elliptic"
	else:
		spectral_class = "parabolic"

	eigenvalues, eigenvectors = _ordered_eigensystem(matrix, spectral_class)
	eigenvector_line_angles = np.full(2, np.nan, dtype=float)
	eigendirections_defined = spectral_class == "hyperbolic"
	if eigendirections_defined:
		for column in range(2):
			vector = eigenvectors[:, column]
			if float(np.max(np.abs(vector.imag))) > 64.0 * np.finfo(float).eps:
				eigendirections_defined = False
				break
			eigenvector_line_angles[column] = line_angle(vector.real)
	if not eigendirections_defined:
		eigenvector_line_angles.fill(np.nan)

	_, singular_values, right_transpose = np.linalg.svd(matrix)
	right_singular_vectors = _canonical_right_singular_vectors(
		right_transpose.T
	)
	singular_gap = float(singular_values[0] - singular_values[1])
	singular_tolerance = tolerance * max(1.0, float(singular_values[0]))
	singular_directions_defined = singular_gap > singular_tolerance
	singular_vector_line_angles = np.full(2, np.nan, dtype=float)
	if singular_directions_defined:
		for column in range(2):
			singular_vector_line_angles[column] = line_angle(
				right_singular_vectors[:, column]
			)

	eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
	eigenvalue_separation = float(
		abs(eigenvalues[1] - eigenvalues[0]) / eigenvalue_scale
	)
	return ParticleJacobianAnalysis(
		jacobian=matrix.copy(),
		trace=trace,
		determinant=determinant,
		discriminant=float(discriminant),
		discriminant_tolerance=float(discriminant_tolerance),
		spectral_class=spectral_class,
		condition_number=float(np.linalg.cond(matrix)),
		spectral_radius=float(np.max(np.abs(eigenvalues))),
		eigenvalue_separation=eigenvalue_separation,
		eigenvector_condition_number=float(np.linalg.cond(eigenvectors)),
		eigendirections_defined=eigendirections_defined,
		eigenvalues=eigenvalues,
		eigenvectors=eigenvectors,
		eigenvector_line_angles=eigenvector_line_angles,
		singular_values=np.asarray(singular_values, dtype=float),
		right_singular_vectors=right_singular_vectors,
		singular_directions_defined=singular_directions_defined,
		singular_vector_line_angles=singular_vector_line_angles,
	)


def particle_jacobian_blocks(
	jacobian: np.ndarray,
	particle_count: int,
) -> np.ndarray:
	"""Extract independent ``[x_i, y_i]`` blocks from component-major layout."""
	if (
		isinstance(particle_count, (bool, np.bool_))
		or not isinstance(particle_count, (int, np.integer))
		or particle_count < 1
	):
		raise ValueError("`particle_count` must be a positive integer.")
	count = int(particle_count)
	matrix = np.asarray(jacobian, dtype=float)
	expected = (2 * count, 2 * count)
	if matrix.shape != expected or not np.all(np.isfinite(matrix)):
		raise ValueError(f"The packed Jacobian must be finite and have shape {expected}.")
	blocks = np.empty((count, 2, 2), dtype=float)
	for particle in range(count):
		indices = (particle, count + particle)
		blocks[particle] = matrix[np.ix_(indices, indices)]
	return blocks


__all__ = [
	"ParticleJacobianAnalysis",
	"SPECTRAL_CLASSES",
	"SpectralClass",
	"analyze_particle_jacobian",
	"line_angle",
	"particle_jacobian_blocks",
]
