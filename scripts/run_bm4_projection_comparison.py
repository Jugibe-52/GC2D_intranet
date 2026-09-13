"""Run and render the reproducible measured-potential BM4 comparison."""

import argparse
from dataclasses import asdict
from pathlib import Path

from diagnostics.bm4_comparison import load_bm4_comparison, save_bm4_comparison
from diagnostics.paths import find_project_root
from potential import load_gc2d_h5_potential
from studies.bm4_projection_comparison import (
	BM4ProjectionComparisonConfig, radial_configuration, run_bm4_projection_comparison,
)
from visualization.bm4_projection_comparison import render_bm4_comparison


def main() -> None:
	"""Keep a duration-specific result directory and support rendering without reruns."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--cycles", type=int, default=20)
	parser.add_argument("--render-only", action="store_true")
	args = parser.parse_args()
	root = find_project_root(Path(__file__))
	output = root / "notebooks/developements/bm4_projection_comparison" / f"{args.cycles}_cycles"
	source = root / "data/potential/V1/PHI_2.h5"
	potential_parameters = dict(B=1.5, characteristic_length=0.06, indx=(0, 1), interpolation_order=3)
	potential = load_gc2d_h5_potential(source, **potential_parameters)
	if args.render_only:
		arrays, metadata = load_bm4_comparison(output / "results.npz")
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
	render_bm4_comparison(potential, arrays, metadata, output, frames=201, fps=10)
	print(f"Report: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
	main()
