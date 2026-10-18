#!/usr/bin/env python3
"""Prepare, execute or visualize a fixed 48-particle Poincare study locally."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support running this repository script without relying on the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studies.poincare_automation import AutomationConfig, METHODS
from studies.poincare_automation_io import prepare_automation, read_automation, run_automation
from visualization.poincare_automation import export_automation_viewer


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	sub = parser.add_subparsers(dest="operation", required=True)
	prepare = sub.add_parser("prepare", help="Save resolved inputs without integrating")
	prepare.add_argument("--rho", nargs="+", type=float, required=True, dest="rhos")
	prepare.add_argument("--method", nargs="+", choices=METHODS, required=True, dest="methods")
	prepare.add_argument("--B", nargs="+", type=float, default=[1.5], dest="magnetic_fields")
	prepare.add_argument("--source", default="data/potential/V1/PHI_2.h5")
	prepare.add_argument("--source-index", type=int, default=15)
	prepare.add_argument("--spatial-rescaling", action=argparse.BooleanOptionalAction, default=True)
	prepare.add_argument("--temporal-rescaling", action=argparse.BooleanOptionalAction, default=True)
	prepare.add_argument("--reference-length", type=float, default=.06)
	prepare.add_argument("--sigma", type=float)
	prepare.add_argument("--interpolation-order", type=int, default=3)
	prepare.add_argument("--cycles", type=int, default=5000)
	prepare.add_argument("--steps-per-cycle", type=int, default=50)
	prepare.add_argument("--coupling-frequency", type=float, default=0.)
	prepare.add_argument("--newton-absolute-tolerance", type=float, default=1e-12)
	prepare.add_argument("--newton-relative-tolerance", type=float, default=1e-11)
	prepare.add_argument("--newton-max-iterations", type=int, default=40)
	prepare.add_argument("--backend", choices=("scipy", "jax"), default="scipy")
	prepare.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
	prepare.add_argument("--device-index", type=int, default=0)
	prepare.add_argument("--cycles-per-frame", type=int, default=25)
	prepare.add_argument("--field-grid-size", type=int, default=64)
	prepare.add_argument("--vector-grid-size", type=int, default=17)
	prepare.add_argument("--output", type=Path)
	for operation in ("run", "view", "status"):
		command = sub.add_parser(operation)
		command.add_argument("directory", type=Path)
	args = vars(parser.parse_args())
	operation = args.pop("operation")
	try:
		if operation == "prepare":
			output = args.pop("output")
			config = AutomationConfig(**args)
			directory = prepare_automation(config, Path(__file__).resolve().parents[1], output)
			print(f"Prepared {len(config.cases())} cases · {config.cycles} cycles · "
				f"{config.steps_per_cycle} steps/cycle · "
				f"{len(config.cases())*config.cycles*config.steps_per_cycle} total steps")
			print(directory)
			return 0
		directory = args["directory"].expanduser().resolve()
		config, _, manifest = read_automation(directory)
		if operation == "status":
			for row in manifest["cases"].values():
				print(row["case"], row["status"], row["error"])
			return 0
		if operation == "run":
			print(f"Running {len(config.cases())} cases at {directory}; "
				f"{config.cycles} cycles, {config.steps_per_cycle} steps/cycle; "
				f"backend={config.backend}, device={config.device}", flush=True)
			manifest = run_automation(directory)
			if any(row["status"] == "complete" for row in manifest["cases"].values()):
				print(export_automation_viewer(directory))
			return int(any(row["status"] != "complete" for row in manifest["cases"].values()))
		print(export_automation_viewer(directory))
		return 0
	except (ValueError, TypeError, OSError, RuntimeError) as exc:
		parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
	raise SystemExit(main())
