"""Dedicated float64 JAX CPU worker for the main reflected campaign."""
from pathlib import Path
import sys
import modal
from execution._modal_worker import execute_payload
from diagnostics.studies.main_campaign import compact_diagnostics
from states.history import States
from diagnostics.studies.campaign_resources import (CPU_CORES, MEMORY_REQUEST_MIB, MEMORY_LIMIT_MIB,
    start_measurement, finish_measurement)

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
    .add_local_file(ROOT / 'constraints.txt', '/tmp/gc2d-constraints.txt', copy=True)
    .pip_install_from_pyproject(str(ROOT / 'pyproject.toml'), optional_dependencies=['jax'],
                               extra_options='-c /tmp/gc2d-constraints.txt')
    .env({'JAX_ENABLE_X64': 'true', 'JAX_PLATFORMS': 'cpu'})
    .add_local_python_source('contracts', 'diagnostics', 'dynamics', 'execution', 'formulations',
                            'states', 'methods', 'potential', 'solvers', 'problem', 'request'))
app = modal.App('gc2d-main-reflected')


@app.function(image=image, cpu=(CPU_CORES, CPU_CORES), memory=(MEMORY_REQUEST_MIB, MEMORY_LIMIT_MIB), timeout=14400,
              retries=0, max_containers=16, scaledown_window=60)
def integrate(payload: bytes) -> tuple[int, str, States]:
    """Compute once; return compact diagnostics without any bucket credentials."""
    started = start_measurement()
    version, digest, solution = execute_payload(payload)
    stride = int(solution.diagnostics['step_count']) // (solution.t.size - 1)
    compact = compact_diagnostics(solution, stride)
    return version, digest, compact.with_diagnostics(finish_measurement(started))
