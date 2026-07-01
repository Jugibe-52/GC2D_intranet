"""Logging helpers for GC2D command-line runs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(processName)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None, log_file: str | Path | None = None) -> None:
	"""Configure root logging once for scripts.

	Environment overrides:
	- GC2D_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR.
	- GC2D_LOG_FILE: optional path for a file log.
	"""
	level_name = (level or os.environ.get("GC2D_LOG_LEVEL") or "INFO").upper()
	log_level = getattr(logging, level_name, None)
	if log_level is None:
		print(f"Invalid GC2D log level {level_name!r}; falling back to INFO.", file=sys.stderr)
		level_name = "INFO"
		log_level = logging.INFO
	file_path = log_file or os.environ.get("GC2D_LOG_FILE")

	handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
	if file_path:
		path = Path(file_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		handlers.append(logging.FileHandler(path, encoding="utf-8"))

	logging.basicConfig(
		level=log_level,
		format=DEFAULT_LOG_FORMAT,
		datefmt=DEFAULT_DATE_FORMAT,
		handlers=handlers,
		force=True,
	)
	logging.getLogger(__name__).debug("Logging configured: level=%s file=%s", level_name, file_path or "stdout only")


def simulation_label(params: dict[str, Any]) -> str:
	"""Return a compact label for log messages about one parameter set."""
	return (
		f"method={params.get('Method', 'unknown')} "
		f"traj={params.get('traj_type', 'unknown')} "
		f"A={params.get('A', 'n/a')} "
		f"rho={params.get('rho', 'n/a')} "
		f"Ntraj={params.get('Ntraj', 'n/a')} "
		f"Tf={params.get('Tf', 'n/a')}"
	)
