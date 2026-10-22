"""One existing Poincare viewer for all fixed-particle automation cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from studies.poincare_automation import PreparedAutomationCase, PROBE_FRACTIONS, PROBE_GAPS
from studies.poincare_automation import prepare_automation_case
from studies.poincare_automation_io import (
	case_key, read_automation, verified_automation_field, load_completed_case,
)
from studies.poincare_gap_probes import GapProbeSeeds
from visualization.poincare_gap_overlay import gap_probe_presentation
from visualization.poincare_rho_sweep import radius_colors
from visualization.poincare_comparison import export_poincare_panel_comparison


if TYPE_CHECKING:
	from diagnostics.persistence import StoredSolution
	from dynamics.gc import GuidingCenterDynamics
	from studies.poincare_automation import AutomationConfig, AutomationField


def particle_presentation() -> dict[str, Any]:
	"""Reuse the original 40-color gradient and eight fixed probe colors."""
	probes: dict[Any, Any] = gap_probe_presentation(GapProbeSeeds(fractions=PROBE_FRACTIONS,
		particle_ids=tuple(range(41, 49)), gap_names=PROBE_GAPS))
	labels = {i+1: f"#{i+1} · arm {i%8+1} · radius {85*i/39:.2f}%" for i in range(40)}
	labels.update(probes["labels"])
	groups = [dict(label=f"Arm {arm+1}", particle_ids=list(range(arm+1, 41, 8))) for arm in range(8)]
	groups += probes["groups"]
	return dict(ids=list(range(1, 49)), colors=radius_colors(40, colormap="turbo",
		color_range=(.025, .975)) + probes["colors"], labels=labels, groups=groups)


def sample_automation_field(prepared: PreparedAutomationCase) -> dict[str, Any]:
	"""Sample H and GC velocity in unit-cell/cycle units; no integration occurs."""
	c: AutomationConfig
	m: dict[str, Any]
	dynamics: GuidingCenterDynamics
	c, m, dynamics = prepared.config, prepared.metadata, prepared.dynamics
	x0: Any
	y0: Any
	length: Any
	x0, y0, length = m["cell_bounds"]
	D = m["runtime_cycle_duration"]
	phases: np.ndarray = np.linspace(0., 1., 51)
	q = (np.arange(c.field_grid_size)+.5)/c.field_grid_size
	xx: np.ndarray
	yy: np.ndarray
	xx, yy = np.meshgrid(x0 + length*q, y0 + length*q)
	vq = (np.arange(c.vector_grid_size)+.5)/c.vector_grid_size
	vx: np.ndarray
	vy: np.ndarray
	vx, vy = np.meshgrid(x0 + length*vq, y0 + length*vq)
	state = np.r_[vx.ravel(), vy.ravel()]
	potential: np.ndarray = np.stack([dynamics.effective_potential.evaluate(phase*D, xx, yy) for phase in phases])
	velocity: np.ndarray = np.stack([dynamics.vector_field(phase*D, state).reshape(2, c.vector_grid_size,
		c.vector_grid_size).transpose(1, 2, 0) for phase in phases])
	potential *= D/length**2
	velocity *= D/length
	for values in (potential, velocity):
		if not np.isfinite(values).all():
			raise ValueError("Viewer fields must be finite.")
		scale: float = max(float(np.max(np.abs(values))), 1e-30)
		np.testing.assert_allclose(values[-1], values[0], rtol=1e-11, atol=1e-13*scale)
		values[-1] = values[0]
	return dict(potential=potential, velocity=velocity, phases=phases, bounds=np.array((0., 0., 1.)),
		units="cell² / cycle; velocity: cell / cycle")


def export_automation_viewer(directory: Path) -> Path:
	"""Validate completed archives and publish one viewer, keeping missing cases unavailable."""
	config: AutomationConfig
	root: Path
	manifest: dict[str, Any]
	config, root, manifest = read_automation(directory)
	if not any(row["status"] == "complete" for row in manifest["cases"].values()):
		raise ValueError("No completed cases are available to visualize.")
	field: AutomationField = verified_automation_field(config, root, manifest)
	presentation: dict[str, Any] = particle_presentation()
	common: dict[str, Any] = dict(particle_ids=presentation["ids"], colors=presentation["colors"])
	only_rho = len(config.methods) == len(config.magnetic_fields) == 1
	datasets: list[dict[str, Any]] = []
	fields: dict[tuple[float, float], dict[str, Any]] = {}
	for case in config.cases():
		key: str = case_key(config, case)
		label: str = repr(case.rho) if only_rho else f"rho={case.rho!r} · B={case.B!r} · {case.method}"
		item: dict[str, Any] = dict(key=key, label=label)
		if manifest["cases"][key]["status"] == "complete":
			prepared: PreparedAutomationCase = prepare_automation_case(config, case, field)
			stored: StoredSolution = load_completed_case(directory, prepared)
			x: np.ndarray
			y: np.ndarray
			x, y = stored.solution.positions()
			xy: np.ndarray = np.stack((x.T, y.T), axis=-1)
			cell: np.ndarray = np.asarray(prepared.metadata["cell_bounds"])
			fractions = ((xy - cell[:2]) % cell[2])/cell[2]
			if (case.rho, case.B) not in fields:
				fields[case.rho, case.B] = sample_automation_field(prepared)
			item["panels"] = [
				dict(common, title="Effective potential and GC velocity", static=True,
					coordinates=fractions[:1], field=fields[case.rho, case.B]),
				dict(common, title=f"{case.method} · rho={case.rho:g} · B={case.B:g}", coordinates=fractions),
			]
			item["note"] = (f"{config.cycles} cycles · {config.steps_per_cycle} steps/cycle · "
				f"rho_m={prepared.metadata['rho_m']:.9g} m · B={case.B:g} · "
				f"physical period={prepared.metadata['physical_period_seconds']:.9g} s")
		datasets.append(item)
	selected: dict[str, Any] = next(d for d in datasets if "panels" in d)
	output = directory / "viewer.html"
	export_poincare_panel_comparison(output, selected["panels"],
		cycles_per_frame=config.cycles_per_frame, first_cycle=0, initial_cycle=0,
		initial_view=(0., 0., 1.), title="Poincare · 48 fixed particles",
		axis_labels=("(R - R0) / L", "(Z - Z0) / L"),
		particle_labels=presentation["labels"], particle_groups=presentation["groups"],
		datasets=datasets, selected_dataset=selected["key"], dataset_label="rho" if only_rho else "Case",
		description="The same 40 star particles and eight fixed probes in every case. "
			"All returns, including cycle zero, are retained. Folding affects display only.")
	return output


__all__ = ["particle_presentation", "sample_automation_field", "export_automation_viewer"]
