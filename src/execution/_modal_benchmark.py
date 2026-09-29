"""Measure complete JAX integrations inside one remotely provisioned CPU worker."""

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter, process_time

import numpy as np

from contracts.result import IntegrationData
from execution._modal_worker import PAYLOAD_VERSION, decode_job
from execution.execution import Execution


def benchmark_payload(payload: bytes, *, cpu_cores: int, repetitions: int) -> tuple[int, str, IntegrationData]:
    """Separate first-call setup/JIT from repeated integrations of the same job.

    The returned NumPy arrays synchronize the device on every invocation. Reuse
    the same physical objects so JAX field bindings and compiled kernels remain
    cached. Provisioning, input decoding and network transfer are not timed here.
    """
    if repetitions < 3 or cpu_cores not in (1, 2, 4):
        raise ValueError("Use at least three repetitions and 1, 2 or 4 CPU cores.")
    problem, method, request, options = decode_job(payload)
    if options.backend != "jax" or options.device != "cpu":
        raise ValueError("This benchmark requires explicit JAX CPU options.")
    import jax
    import jaxlib
    import scipy

    executor = Execution()
    start = perf_counter()
    first = executor.run(problem, method, request, options=options)
    first_seconds = perf_counter() - start
    wall, cpu = [], []
    for _ in range(repetitions):
        start, cpu_start = perf_counter(), process_time()
        data = executor.run(problem, method, request, options=options)
        cpu.append(process_time() - cpu_start)
        wall.append(perf_counter() - start)
        np.testing.assert_array_equal(data.t, first.t)
        np.testing.assert_allclose(data.states, first.states, rtol=1e-12, atol=1e-12)
    cpu_model = platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    quota = Path('/sys/fs/cgroup/cpu.max')
    environment = {
        "cpu_model": cpu_model, "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "jax_enable_x64": bool(jax.config.read('jax_enable_x64')),
        "visible_logical_cpus": os.cpu_count(),
        "affinity_logical_cpus": len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        "cpu_max": quota.read_text().strip() if quota.exists() else "unavailable",
        "threads": {name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')},
    }
    diagnostics = dict(data.diagnostics)
    diagnostics.update(
        benchmark_cpu_cores=cpu_cores, benchmark_repetitions=repetitions,
        benchmark_first_seconds=first_seconds, benchmark_wall_seconds=np.asarray(wall),
        benchmark_process_seconds=np.asarray(cpu), benchmark_environment=json.dumps(environment),
    )
    return PAYLOAD_VERSION, sha256(payload).hexdigest(), IntegrationData(data.t, data.states, diagnostics)


__all__: list[str] = []
