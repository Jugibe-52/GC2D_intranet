"""Run and render the reproducible measured-potential BM4 comparison."""

import argparse
from dataclasses import asdict
from pathlib import Path

from diagnostics.bm4_comparison import load_bm4_comparison, save_bm4_comparison
from diagnostics import load_reference_trajectory
from diagnostics.paths import find_project_root
from potential import load_gc2d_h5_potential
from studies.bm4_projection_comparison import (
	BM4ProjectionComparisonConfig, radial_configuration, run_bm4_projection_comparison,
	audit_saved_reference_refinement,
)
from studies.initial_conditions import radial_gc_configuration, domain_center
from visualization.bm4_projection_comparison import render_bm4_comparison


def main() -> None:
	"""Keep a duration-specific result directory and support rendering without reruns."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--cycles", type=int, default=20)
	parser.add_argument("--render-only", action="store_true")
	parser.add_argument("--compute-only", action="store_true")
	parser.add_argument("--standard-reference", action="store_true",
		help="Reuse the saved standard radial reference through normalized time 35.")
	args = parser.parse_args()
	root = find_project_root(Path(__file__))
	output = root / "notebooks/developements/bm4_projection_comparison" / f"{args.cycles}_cycles"
	if args.standard_reference:
		output = root / "notebooks/developements/bm4_projection_comparison/standard_t35"
	source = root / "data/potential/V1/PHI_2.h5"
	potential_parameters = dict(B=1.5, characteristic_length=0.06, indx=(0, 1), interpolation_order=3)
	potential = load_gc2d_h5_potential(source, **potential_parameters)
	if args.render_only:
		arrays, metadata = load_bm4_comparison(output / "results.npz")
	elif args.standard_reference:
		potential_parameters["spatial_normalization"] = "characteristic_length"
		reference = load_reference_trajectory(root / "data/trajectory/h5_three_radial_dop853_t35")
		coarse = load_reference_trajectory(root / "outputs/developements/accuracy/h5_three_radial/v2")
		config = BM4ProjectionComparisonConfig(t_span=(0., 35.), progress=True,
			reference_relative_tolerance=5e-13, reference_absolute_tolerance=5e-15,
			reference_maximum_step=.0025, audit_relative_tolerance=5e-14,
			audit_absolute_tolerance=5e-16, audit_maximum_step=.00125)
		configuration = radial_gc_configuration(potential, radial_fractions=(.1, .2, .3), angle=0.)
		x, y = configuration.positions(configuration.initial_state)
		initial_metadata = {"geometry": "one radial segment", "center": list(domain_center(potential)),
			"radial_fractions": [.1, .2, .3], "angle_radians": 0., "x": x.tolist(), "y": y.tolist()}
		potential_metadata = {"source": str(source), **potential_parameters}
		checks, refinement = audit_saved_reference_refinement(potential, configuration, reference, coarse,
			config=config, potential_metadata=potential_metadata, initial_condition_metadata=initial_metadata)
		print(f"Saved-reference refinement verified: {refinement}", flush=True)
		arrays, summary = run_bm4_projection_comparison(potential, configuration, config=config,
			saved_reference=reference, potential_metadata=potential_metadata, initial_condition_metadata=initial_metadata)
		arrays.update(checks)
		summary["reference_validation"] = refinement
		metadata = {"config": asdict(config), "source": str(source), "potential_parameters": potential_parameters,
			"initial_geometry": initial_metadata, "reference_metadata": dict(reference.metadata),
			"reference_directory": str(reference.paths.directory),
			"protocol_scope": "Two BM4 methods over normalized time [0,35], not 35 oscillations. Standard radial fractions of cell period: (0.1,0.2,0.3). Saved references are reused and verified by refinement.",
			"summary": summary}
		save_bm4_comparison(output / "results.npz", arrays, metadata)
		print(f"Saved numerical results to {output}", flush=True)
	else:
		config = BM4ProjectionComparisonConfig(t_span=(0.0, float(args.cycles)), progress=True)
		configuration = radial_configuration(potential, radius_fractions=(0.1, 0.25, 0.4), angle=0.0)
		arrays, summary = run_bm4_projection_comparison(potential, configuration, config=config)
		metadata = {
			"config": asdict(config), "source": str(source), "potential_parameters": potential_parameters,
			"initial_geometry": {"radius_fractions_of_half_width": [0.1, 0.25, 0.4], "angle_radians": 0.0},
			"protocol_scope": "Only the two requested BM4 methods; radial starts replace the standard spatial sample. Twenty cycles give an initial comparison; use --cycles 200 for the standard long horizon.",
			"summary": summary,
		}
		save_bm4_comparison(output / "results.npz", arrays, metadata)
		print(f"Saved numerical results to {output}", flush=True)
	if not args.compute_only:
		render_bm4_comparison(potential, arrays, metadata, output, frames=201, fps=10)
		print(f"Report: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
	main()
