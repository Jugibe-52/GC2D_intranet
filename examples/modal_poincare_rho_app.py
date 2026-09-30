"""Deploy one JAX CPU worker for the dimensional Poincare rho sweep."""

from pathlib import Path
import sys
from time import perf_counter

import modal

from contracts.result import IntegrationData
from execution._modal_worker import execute_payload

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
app = modal.App("gc2d-poincare-rho-sweep")


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=7200,
              retries=0, max_containers=1, scaledown_window=60)
def integrate(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Run a single float64 integration, returning results to the local caller."""
    import numpy as np
    start = perf_counter()
    version, digest, data = execute_payload(payload)
    # Transfer cycle observations and scalar diagnostics, without per-step arrays.
    diagnostics = {k: v for k, v in data.diagnostics.items() if not isinstance(v, np.ndarray)}
    diagnostics['worker_wall_seconds'] = perf_counter() - start
    return version, digest, IntegrationData(data.t, data.states, diagnostics)
