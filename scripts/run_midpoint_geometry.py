#!/usr/bin/env python3
"""Run the fixed midpoint geometry protocol, preserving a local result archive."""
from __future__ import annotations
import argparse
from pathlib import Path
from studies.midpoint_geometry import run_study


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output', type=Path, default=Path('outputs/midpoint_geometry/midpoint-geometry-20261118-rho0015'))
	parser.add_argument('--skip-accumulation', action='store_true')
	args = parser.parse_args()
	root = Path(__file__).resolve().parents[1]
	run_study(root, args.output, skip_accumulation=args.skip_accumulation)


if __name__ == '__main__':
	main()
