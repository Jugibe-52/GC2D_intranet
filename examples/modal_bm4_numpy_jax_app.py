"""Matched NumPy/SciPy and JAX CPU benchmark for thesis section 4.2."""
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
    .add_local_file(ROOT / 'constraints.txt', '/tmp/gc2d-constraints.txt', copy=True)
    .pip_install_from_pyproject(str(ROOT / 'pyproject.toml'), optional_dependencies=['jax'],
                               extra_options='-c /tmp/gc2d-constraints.txt')
    .env({'JAX_ENABLE_X64': 'true', 'JAX_PLATFORMS': 'cpu',
          'JAX_ENABLE_COMPILATION_CACHE': 'false'})
    .add_local_python_source('contracts', 'diagnostics', 'dynamics', 'execution', 'formulations',
                            'states', 'methods', 'potential', 'solvers', 'problem', 'request', 'studies', 'simulation'))
app = modal.App('gc2d-bm4-numpy-jax')


@app.function(image=image, cpu=(2., 2.), memory=(4096, 8192), timeout=86400,
              retries=0, max_containers=6, scaledown_window=60)
def benchmark(payload: bytes, repetitions: int = 3):
    """Keep paired backends sequential inside each isolated CPU container."""
    from studies.bm4_numpy_jax import benchmark_pair
    return benchmark_pair(payload, repetitions)
