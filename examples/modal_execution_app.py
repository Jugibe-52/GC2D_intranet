"""Deploy the CPU integration endpoint with `modal deploy examples/modal_execution_app.py`.

Edit server resources here independently of Execution_Modal. The image contains
project code and numerical dependencies, without notebooks, source data, bucket
credentials or results. Deployment is an explicit operation.
"""

from pathlib import Path
import sys

import modal

from execution._modal_worker import execute_payload
from contracts.result import IntegrationData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .add_local_file(PROJECT_ROOT / "constraints.txt", "/tmp/gc2d-constraints.txt", copy=True)
    .pip_install_from_pyproject(str(PROJECT_ROOT / "pyproject.toml"),
                               extra_options="-c /tmp/gc2d-constraints.txt")
    .add_local_python_source("contracts", "dynamics", "execution", "formulations",
                             "initial_conditions", "integration", "methods", "potential", "solution")
)
app = modal.App("gc2d-execution")


@app.function(image=image, timeout=3600, retries=0)
def integrate(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Return a complete NumPy result; persistence happens on the caller."""
    return execute_payload(payload)
