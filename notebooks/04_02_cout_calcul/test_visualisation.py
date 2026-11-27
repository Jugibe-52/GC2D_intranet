"""Offline checks against the saved scientific archive, without integration."""
from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
DIRECTORY = Path(__file__).resolve().parent
ARCHIVE = DIRECTORY.parents[1] / "outputs/studies/method_timings/jax-20261118-rho0015-c500"


def load_definitions() -> dict:
	"""Load only the portable notebook's definitions, with no download or output."""
	source = ast.parse((DIRECTORY / "visualisation.py").read_text(encoding="utf-8"))
	constants = {"PUBLIC_DATA_BASE_URL", "EXPECTED_MANIFEST_SHA256", "STUDY_PREFIX", "RUN_ID", "METHODS", "LABELS", "COLORS"}
	source.body = [node for node in source.body
		if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
		or isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
		and node.targets[0].id in constants]
	namespace = {}
	exec(compile(source, "visualisation.py", "exec"), namespace)
	return namespace


class PortableTimingTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		if not (ARCHIVE / "timings.json").is_file():
			raise unittest.SkipTest("The saved timing archive is unavailable on this computer.")
		cls.objects = {name: (ARCHIVE / name).read_bytes() for name in ("study.json", "timings.json")}
		cls.study = json.loads(cls.objects["study.json"])
		cls.timings = json.loads(cls.objects["timings.json"])
		cls.baseline = json.loads((DIRECTORY.parents[2] / "TFM_redaccion/figuras/comparacion_tiempos_metodos_fr.json").read_text())
		manifest = dict(schema_version=1, study_prefix=cls.study["study_prefix"], run_id=cls.study["run_id"],
			files={name: dict(bytes=len(data), sha256=sha256(data).hexdigest()) for name, data in cls.objects.items()})
		cls.objects["manifest.json"] = json.dumps(manifest).encode()

	def setUp(self) -> None:
		self.namespace = load_definitions()
		self.namespace["EXPECTED_MANIFEST_SHA256"] = sha256(self.objects["manifest.json"]).hexdigest()
		self.namespace["download_bytes"] = lambda name: self.objects[name]

	def test_statistics_equal_archived_figure_exactly(self) -> None:
		study, timings, manifest = self.namespace["load_public_data"]()
		summary = self.namespace["summarize_timings"](study, timings)
		for key in ("rows", "fastest_method", "environment", "call_id", "repeated_states_identical"):
			self.assertEqual(summary[key], self.baseline[key])
		self.assertEqual(manifest["files"]["timings.json"]["sha256"], self.baseline["timings_sha256"])

	def test_changed_manifest_is_rejected(self) -> None:
		self.namespace["EXPECTED_MANIFEST_SHA256"] = "0" * 64
		with self.assertRaisesRegex(ValueError, "pinned version"):
			self.namespace["load_public_data"]()

	def test_corrupt_or_truncated_data_is_rejected(self) -> None:
		for data in (self.objects["timings.json"][:-1], self.objects["timings.json"].replace(b'6.037879243', b'0.037879243')):
			with self.subTest(length=len(data)):
				self.namespace["download_bytes"] = lambda name: data if name == "timings.json" else self.objects[name]
				with self.assertRaisesRegex(ValueError, "size or SHA-256"):
					self.namespace["load_public_data"]()

	def test_invalid_recorded_protocol_is_rejected(self) -> None:
		for key, value in (("phase", "warm"), ("position", 7), ("step_count", 19999),
				("integration_cache_hits", 1), ("seconds", -1), ("seconds", float("nan"))):
			with self.subTest(key=key, value=value):
				timings = deepcopy(self.timings)
				timings["measurements"][0][key] = value
				with self.assertRaises(ValueError):
					self.namespace["summarize_timings"](self.study, timings)
		for change in ("missing", "duplicated"):
			with self.subTest(change=change):
				timings = deepcopy(self.timings)
				if change == "missing":
					timings["measurements"].pop()
				else:
					timings["measurements"][-1] = timings["measurements"][0]
				with self.assertRaises(ValueError):
					self.namespace["summarize_timings"](self.study, timings)

	def test_figure_retains_french_axes_and_all_measurements(self) -> None:
		summary = self.namespace["summarize_timings"](self.study, self.timings)
		figure = self.namespace["plot_timings"](summary)
		try:
			axis = figure.axes[0]
			self.assertEqual(axis.get_xlabel(), "Temps de calcul (s)")
			self.assertEqual(axis.get_xscale(), "log")
			self.assertEqual([item.get_text() for item in axis.get_yticklabels()], list(self.namespace["LABELS"]))
			self.assertEqual(axis.child_axes[0].get_xlabel(), "Temps / médiane de RK2")
			self.assertEqual(len(axis.containers), 8)
		finally:
			self.namespace["plt"].close(figure)


if __name__ == "__main__":
	unittest.main()
