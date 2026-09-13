"""Small end-to-end checks of the BM4 comparison and persistence contracts."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.bm4_comparison import load_bm4_comparison, save_bm4_comparison
from potential import Potential
from studies.bm4_projection_comparison import (
	BM4ProjectionComparisonConfig, radial_configuration, run_bm4_projection_comparison,
)
from visualization.bm4_projection_comparison import plot_bm4_comparison, write_bm4_report


class BM4ProjectionComparisonTests(unittest.TestCase):
	def test_comparison_retains_signed_mu_and_excludes_explicit_nonlinear_work(self) -> None:
		potential = Potential.random(A=.05, M=2, nx=12, ny=12, seed=7, interpolation_order=3)
		configuration = radial_configuration(potential)
		config = BM4ProjectionComparisonConfig(t_span=(0., .2), integration_step=.1,
			timing_warmups=0, timing_repeats=3)
		arrays, summary = run_bm4_projection_comparison(potential, configuration, config=config)
		self.assertEqual(arrays["BM4Implicit.mu"].shape, (6, 2))
		self.assertNotIn("BM4Midpoint.mu", arrays)
		self.assertNotIn("BM4Midpoint.nonlinear_iterations", arrays)
		for name in ("BM4Midpoint", "BM4Implicit"):
			self.assertEqual(arrays[f"{name}.states"].shape, (6, 3))
			self.assertEqual(arrays[f"{name}.runtime_samples"].size, 3)
			np.testing.assert_array_equal(arrays[f"{name}.states"][:, 0], arrays["initial_state"])
		metadata = {"config": asdict(config), "summary": summary}
		with tempfile.TemporaryDirectory() as temporary:
			path = Path(temporary) / "results.npz"
			save_bm4_comparison(path, arrays, metadata)
			loaded, stored_metadata = load_bm4_comparison(path)
			np.testing.assert_array_equal(loaded["BM4Implicit.mu"], arrays["BM4Implicit.mu"])
			self.assertEqual(stored_metadata["summary"], summary)
			write_bm4_report(loaded, stored_metadata, temporary)
			self.assertTrue((Path(temporary) / "report.html").is_file())
		figures = plot_bm4_comparison(arrays, metadata)
		self.assertEqual(len(figures), 5)
		for figure in figures.values():
			figure.canvas.draw()
			plt.close(figure)


if __name__ == "__main__":
	unittest.main()
