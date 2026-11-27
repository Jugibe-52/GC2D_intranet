"""One independently recoverable projection case on the reader's Modal account."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy==1.26.4', 'scipy==1.17.1', 'h5py==3.16.0',
        'threadpoolctl==3.6.0', 'jax==0.7.1', 'jaxlib==0.7.1')
    .env({'JAX_ENABLE_X64': 'true', 'JAX_PLATFORMS': 'cpu',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    .add_local_dir(ROOT / 'kit', '/opt/gc2d')
    .add_local_python_source('calculation_runtime'))
app = modal.App('gc2d-colab-projection-correction')


@app.function(image=image, cpu=(2., 2.), memory=(4096, 12288), timeout=14400,
    retries=0, max_containers=2, scaledown_window=60)
def calculate(case, settings, protocol_sha256):
    from calculation_runtime import calculate_case
    return calculate_case(Path('/opt/gc2d'), case, settings, protocol_sha256)
