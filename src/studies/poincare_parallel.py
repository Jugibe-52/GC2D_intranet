"""Fixed transverse neighbours of nine archived regular Poincare curves."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dynamics.gc import GuidingCenterDynamics
from problem import InitialValueProblem
from studies.horizontal_transport_reproduction import prepare_horizontal_reproduction
from studies.poincare_rho_sweep import PreparedRhoStar

PARENTS = (1, 2, 41, 42, 43, 44, 45, 47, 48)
OFFSETS = (-0.02, -0.04, 0.02, 0.04)
SIDES = ("inner-near", "inner-far", "outer-near", "outer-far")
METHOD_SLUGS = {"BM4Midpoint": "bm4_midpoint", "BM4Implicit": "bm4_implicit",
	"RK4": "rk4", "GaussLegendre4": "gauss_legendre"}


def inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
	"""Classify a point away from the sampled boundary by ray intersections."""
	a: np.ndarray = polygon
	b: np.ndarray = np.roll(a, -1, axis=0)
	mask: np.ndarray = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
	v: np.ndarray = a[mask]
	w: np.ndarray = b[mask]
	crossings: np.ndarray = point[0] < (w[:, 0]-v[:, 0]) * (point[1]-v[:, 1]) / (w[:, 1]-v[:, 1]) + v[:, 0]
	return bool(np.count_nonzero(crossings) % 2)


def normal_geometry(positions: np.ndarray, *, component_stride: int = 1,
	neighbours: int = 40) -> dict[str, Any]:
	"""Fit a local tangent and measure the curve width along its outward normal.

	Input coordinates are cell fractions at integer forcing cycles. A stride
	selects the initial island of a verified island chain. Periodic lifting is
	relative to the initial point and is valid for these compact components.
	"""
	points: np.ndarray = np.asarray(positions, dtype=float)[::component_stride].copy()
	if points.ndim != 2 or points.shape[1] != 2 or len(points) <= neighbours or not np.isfinite(points).all():
		raise ValueError("A curve needs finite two-dimensional samples and enough neighbours.")
	origin: np.ndarray = points[0].copy()
	points = origin + (points-origin+0.5) % 1.0 - 0.5
	if np.any(np.ptp(points, axis=0) >= 0.5):
		raise ValueError("The curve is not a compact component in the chosen periodic lift.")
	center: np.ndarray = points.mean(axis=0)
	delta: np.ndarray = points-origin
	local: np.ndarray = delta[np.argsort(np.linalg.norm(delta, axis=1))[1:neighbours+1]]
	eigenvalues: np.ndarray
	eigenvectors: np.ndarray
	eigenvalues, eigenvectors = np.linalg.eigh(local.T @ local)
	if eigenvalues[-1] <= 0 or eigenvalues[0]/eigenvalues[-1] > 0.02:
		raise ValueError("Saved returns do not determine a sufficiently thin local curve.")
	tangent: np.ndarray = eigenvectors[:, -1]
	normal: np.ndarray = np.array((-tangent[1], tangent[0]))
	if np.dot(normal, origin-center) < 0:
		normal = -normal
	width: float = float(np.ptp(points @ normal))
	angles: np.ndarray = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
	polygon: np.ndarray = points[np.argsort(angles)]
	seeds: np.ndarray = origin + np.asarray(OFFSETS)[:, None] * width * normal
	classification: list[bool] = [inside_polygon(seed, polygon) for seed in seeds]
	if classification != [True, True, False, False]:
		raise ValueError("Normal offsets do not lie on the expected sides of the reconstructed curve.")
	if np.any(seeds < 0) or np.any(seeds >= 1):
		raise ValueError("These reference seeds must lie inside the displayed cell.")
	return dict(initial=origin.tolist(), normal=normal.tolist(), width_fraction=width,
		fractions=seeds.tolist(), inside=classification, component_stride=component_stride,
		fit_variance_ratio=float(eigenvalues[0]/eigenvalues[-1]), neighbours=neighbours)


def load_returns(runs: Path, method: str, rho_index: int, group: str) -> tuple[dict[str, Any], np.ndarray]:
	"""Read a fingerprint-checked archive without importing or reintegrating it."""
	folder: Path = runs / f"{METHOD_SLUGS[method]}_{rho_index:02d}_{group}"
	manifest: dict[str, Any] = json.loads((folder/"manifest.json").read_text())
	for name in ("metadata.json", "solution.npz"):
		if sha256((folder/name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
			raise ValueError(f"Archive fingerprint mismatch: {folder/name}")
	metadata: dict[str, Any] = json.loads((folder/"metadata.json").read_text())
	with np.load(folder/"solution.npz", allow_pickle=False) as archive:
		positions: np.ndarray = archive[metadata["diagnostics"]["cycle_positions_cell_fraction"]["array"]]
	return metadata["experiment"], positions


def build_seed_manifest(runs: Path) -> dict[str, Any]:
	"""Persist the proposed geometry, IDs and independent method checks."""
	curves: dict[tuple[str, int], np.ndarray] = {}
	cell: np.ndarray = np.empty(0)
	for method in METHOD_SLUGS:
		for group in ("star", "probes"):
			metadata: dict[str, Any]
			positions: np.ndarray
			metadata, positions = load_returns(runs, method, 6, group)
			cell = np.asarray(metadata["cell_bounds_m"])
			for index, parent in enumerate(metadata["particle_id"]):
				if parent in PARENTS:
					curves[(method, parent)] = positions[:, index]
	families: list[dict[str, Any]] = []
	particles: list[dict[str, Any]] = []
	for index, parent in enumerate(PARENTS):
		stride: int = 7 if parent == 2 else 1
		geometry: dict[str, Any] = normal_geometry(curves[("GaussLegendre4", parent)], component_stride=stride)
		normal: np.ndarray = np.asarray(geometry["normal"])
		checks: list[dict[str, Any]] = []
		for method in METHOD_SLUGS:
			other: dict[str, Any] = normal_geometry(curves[(method, parent)], component_stride=stride)
			angle: float = float(np.degrees(np.arccos(np.clip(normal @ np.asarray(other["normal"]), -1, 1))))
			if angle > 2:
				raise ValueError(f"Method-dependent normal direction for particle {parent}.")
			checks.append(dict(method=method, normal_angle_degrees=angle,
				width_relative_difference=other["width_fraction"]/geometry["width_fraction"]-1))
		new_ids: list[int] = list(range(49+4*index, 53+4*index))
		families.append(dict(parent=parent, ids=[parent, *new_ids], geometry=geometry, method_checks=checks))
		for pid, side, fraction, xy in zip(new_ids, SIDES, OFFSETS, geometry["fractions"], strict=True):
			particles.append(dict(id=pid, parent=parent, side=side, signed_width_fraction=fraction,
				distance_m=abs(fraction)*geometry["width_fraction"]*cell[2],
				cell_fraction=xy, position_m=(cell[:2]+cell[2]*np.asarray(xy)).tolist()))
	return dict(schema_version=1, reference_rho=0.3, reference_method="GaussLegendre4",
		cell_bounds_m=cell.tolist(), families=families, particles=particles)


def prepare_parallel(source: Path, archived_metadata: Mapping[str, Any], seeds: Mapping[str, Any],
	*, cycles: int = 5000, steps_per_cycle: int = 50) -> PreparedRhoStar:
	"""Reuse the archived dynamics exactly and replace only initial positions."""
	prepared: PreparedRhoStar = prepare_horizontal_reproduction(source, archived_metadata,
		cycles=cycles, steps_per_cycle=steps_per_cycle)
	positions: np.ndarray = np.asarray([p["position_m"] for p in seeds["particles"]], dtype=float)
	ids: np.ndarray = np.asarray([p["id"] for p in seeds["particles"]], dtype=int)
	if positions.shape != (36, 2) or not np.isfinite(positions).all() or not np.array_equal(ids, np.arange(49, 85)):
		raise ValueError("The accepted study requires exactly the fixed 36 seeds numbered 49-84.")
	if not np.array_equal(seeds["cell_bounds_m"], prepared.metadata["cell_bounds_m"]):
		raise ValueError("Seed and field geometries differ.")
	initial: np.ndarray = GuidingCenterDynamics.coordinates.pack_components(positions[:, 0], positions[:, 1])
	problem: InitialValueProblem = InitialValueProblem(prepared.problem.dynamics, initial,
		layout=GuidingCenterDynamics.coordinates)
	# The reused config validates a radial-star arm count; explicit seeds do
	# not use it. Four is a valid divisor of 36 and does not alter the field.
	config: Any = replace(prepared.config, particles=36, arms=4)
	metadata: dict[str, Any] = dict(prepared.metadata)
	for key in ("arm_id", "initial_radii", "initial_radii_m", "initial_radius_fraction", "star_center", "star_center_m", "cell_radius_m"):
		metadata.pop(key, None)
	metadata.update(study="poincare_parallel_families", config=asdict(config), particle_id=ids,
		initial_positions=positions, initial_positions_m=positions,
		initial_positions_cell_fraction=np.asarray([p["cell_fraction"] for p in seeds["particles"]]),
		seed_manifest=seeds, particle_order="persistent IDs 49-84, grouped by original parent",
		initial_geometry="fixed normal offsets selected at rho=0.30",
		initial_geometry_definition="seed_manifest positions; star arm, angle and radius controls are unused",
		historical_reproduction={**metadata["historical_reproduction"],
			"archived_initial_positions_exact": False, "grouping": "thirty_six_parallel_neighbours"})
	return replace(prepared, problem=problem, config=config, metadata=metadata)
