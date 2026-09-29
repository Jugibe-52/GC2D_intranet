"""Deploy matched 1/2/4-core JAX CPU workers for complete integration benchmarks."""

from pathlib import Path
import sys

import modal

from contracts.result import IntegrationData
from execution._modal_benchmark import benchmark_payload


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
app = modal.App("gc2d-jax-cpu-comparison")
REPETITIONS = 3


@app.function(image=image, cpu=(1., 1.), memory=(2048, 4096), timeout=600,
              retries=0, max_containers=1, scaledown_window=60)
def integrate_cpu_1(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Run the complete job with a one-core CPU request and limit."""
    return benchmark_payload(payload, cpu_cores=1, repetitions=REPETITIONS)


@app.function(image=image, cpu=(2., 2.), memory=(2048, 4096), timeout=600,
              retries=0, max_containers=1, scaledown_window=60)
def integrate_cpu_2(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Run the same job with a two-core CPU request and limit."""
    return benchmark_payload(payload, cpu_cores=2, repetitions=REPETITIONS)


@app.function(image=image, cpu=(4., 4.), memory=(2048, 4096), timeout=600,
              retries=0, max_containers=1, scaledown_window=60)
def integrate_cpu_4(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Run the same job with a four-core CPU request and limit."""
    return benchmark_payload(payload, cpu_cores=4, repetitions=REPETITIONS)
