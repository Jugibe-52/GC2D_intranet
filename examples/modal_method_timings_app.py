"""Single-container JAX CPU benchmark for thesis section 4.3."""
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
    .add_local_file(ROOT / 'constraints.txt', '/tmp/gc2d-constraints.txt', copy=True)
    .pip_install_from_pyproject(str(ROOT / 'pyproject.toml'), optional_dependencies=['jax'],
                               extra_options='-c /tmp/gc2d-constraints.txt')
    .env({'JAX_ENABLE_X64': 'true', 'JAX_PLATFORMS': 'cpu',
          'JAX_ENABLE_COMPILATION_CACHE': 'false', 'OMP_NUM_THREADS': '2',
          'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    .add_local_python_source('contracts', 'diagnostics', 'dynamics', 'execution', 'formulations',
                            'states', 'methods', 'potential', 'solvers', 'problem', 'request', 'studies', 'simulation'))
app = modal.App('gc2d-method-timings')


@app.function(image=image, cpu=(2., 2.), memory=(4096, 8192), timeout=14400,
              retries=0, max_containers=1, scaledown_window=60)
def benchmark(payloads: dict[str, bytes], repetitions: int = 3):
    """All methods share the worker and run without competing integrations."""
    from studies.method_timings import benchmark_methods
    return benchmark_methods(payloads, repetitions)
