"""Deploy at most 44 concurrent float64 JAX CPU Poincare integrations."""

from pathlib import Path
import sys

import modal

from contracts.result import IntegrationData
from execution._modal_worker import execute_cycle_payload


ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    .add_local_file(ROOT / "constraints.txt", "/tmp/gc2d-constraints.txt", copy=True)
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["jax"],
                               extra_options="-c /tmp/gc2d-constraints.txt")
    .env({"JAX_ENABLE_X64": "true", "JAX_PLATFORMS": "cpu"})
    .add_local_python_source("contracts", "dynamics", "execution", "formulations",
                             "initial_conditions", "integration", "methods", "potential", "solution")
)
app = modal.App("gc2d-poincare-methods")


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=7200,
              retries=0, max_containers=44, scaledown_window=60)
def integrate(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Execute one complete integration; the local caller publishes its archive."""
    return execute_cycle_payload(payload)
