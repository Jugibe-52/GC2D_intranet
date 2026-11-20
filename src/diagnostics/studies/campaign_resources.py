"""Measured worker resources and explicitly estimated Modal compute charges."""
from __future__ import annotations

import os
import platform
from pathlib import Path
import resource
from time import perf_counter, process_time, time
from typing import Any

# Public standard Functions list prices, checked on 2026-11-15. These are not
# workspace billing records and exclude credits, taxes, startup and idle time.
CPU_USD_PER_CORE_SECOND = .0000131
MEMORY_USD_PER_GIB_SECOND = .00000222
CPU_CORES = 2.
MEMORY_REQUEST_MIB = 4096
MEMORY_LIMIT_MIB = 8192


def start_measurement() -> tuple[float, float, float]:
	return perf_counter(), process_time(), time()


def finish_measurement(start: tuple[float, float, float]) -> dict[str, Any]:
	"""Separate allocated cores from visible logical CPUs and RSS from billing."""
	import jax
	import numpy
	import scipy
	elapsed: float = perf_counter() - start[0]
	model: str = platform.processor() or 'unknown'
	cpuinfo: Path = Path('/proc/cpuinfo')
	if cpuinfo.exists():
		model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
			if line.startswith('model name')), model)
	result: dict[str, Any] = dict(worker_wall_seconds=elapsed,
		worker_process_cpu_seconds=process_time()-start[1],
		worker_started_unix_seconds=start[2], worker_finished_unix_seconds=time(),
		worker_timing_scope='decode, preparation, JAX compilation, integration and diagnostics; excludes container startup and transfer',
		worker_cpu_requested_physical_cores=CPU_CORES, worker_cpu_limit_physical_cores=CPU_CORES,
		worker_visible_logical_cpus=os.cpu_count() or 0,
		worker_memory_requested_mib=MEMORY_REQUEST_MIB, worker_memory_limit_mib=MEMORY_LIMIT_MIB,
		worker_process_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
		worker_cpu_model=model, worker_python=platform.python_version(), worker_jax=jax.__version__,
		worker_numpy=numpy.__version__, worker_scipy=scipy.__version__,
		cost_currency='USD', cost_tariff_date='2026-11-15', cost_tariff_source='https://modal.com/pricing',
		cost_cpu_usd_per_core_second=CPU_USD_PER_CORE_SECOND,
		cost_memory_usd_per_gib_second=MEMORY_USD_PER_GIB_SECOND,
		cost_estimated_requested_resources_usd=elapsed*(CPU_CORES*CPU_USD_PER_CORE_SECOND+MEMORY_REQUEST_MIB/1024*MEMORY_USD_PER_GIB_SECOND),
		cost_estimated_limit_resources_usd=elapsed*(CPU_CORES*CPU_USD_PER_CORE_SECOND+MEMORY_LIMIT_MIB/1024*MEMORY_USD_PER_GIB_SECOND),
		cost_estimate_scope='measured worker interval only; resource-request and resource-limit scenarios; not an invoice',
		cost_billed_usd_status='unavailable in worker result; reconcile with Modal billing separately')
	for label, filename in (('worker_cgroup_cpu_max', '/sys/fs/cgroup/cpu.max'),
			('worker_cgroup_memory_max', '/sys/fs/cgroup/memory.max')):
		path: Path = Path(filename)
		result[label] = path.read_text().strip() if path.exists() else 'unavailable'
	return result


__all__ = ['CPU_CORES', 'MEMORY_REQUEST_MIB', 'MEMORY_LIMIT_MIB', 'start_measurement', 'finish_measurement']
