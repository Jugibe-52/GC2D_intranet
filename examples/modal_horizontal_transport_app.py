"""Bounded JAX CPU workers for the accepted horizontal-transport repetition."""
from pathlib import Path
import sys

import modal

from execution._modal_worker import execute_cycle_payload
from states.history import States

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    .add_local_file(ROOT / "constraints.txt", "/tmp/gc2d-constraints.txt", copy=True)
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["jax"],
                               extra_options="-c /tmp/gc2d-constraints.txt")
    .env({"JAX_ENABLE_X64": "true", "JAX_PLATFORMS": "cpu"})
    .add_local_python_source("contracts", "diagnostics", "dynamics", "execution", "formulations",
                             "states", "methods", "potential", "solvers", "problem", "request")
)
app = modal.App("gc2d-horizontal-transport")


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=2400,
              retries=0, max_containers=44, scaledown_window=5)
def integrate(payload: bytes) -> tuple[int, str, States]:
    """One original star/probe case; finite timeout and no automatic retry."""
    return execute_cycle_payload(payload)


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=600,
              retries=0, max_containers=4, scaledown_window=5)
def benchmark(payload: bytes) -> tuple[int, str, States]:
    """Short validation case with a separate lower time/concurrency ceiling."""
    return execute_cycle_payload(payload)
