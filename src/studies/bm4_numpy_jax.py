"""Paired, synchronized CPU timings for the selected methods in section 4.2."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import platform
from time import perf_counter, process_time
from typing import Any

import numpy as np

from execution._modal_worker import decode_job
from execution.execution import Execution
from execution.options import ExecutionOptions
from states.history import States


def benchmark_pair(payload: bytes, repetitions: int = 3) -> dict[str, Any]:
	"""Measure both libraries on one worker, separating cold and cached calls.

	Input restoration and all audits are outside the measured executor interval.
	States contains NumPy arrays, so returning from run synchronizes JAX. Retain
	one trajectory per backend and check every repeated trajectory bit for bit.
	"""
	if type(repetitions) is not int or repetitions < 3:
		raise ValueError('At least three warmed repetitions are required.')
	import jax
	import jaxlib
	import scipy
	from execution._jax_fixed import _compiled_integrator

	problem, method, options, observers = decode_job(payload)
	if (type(method).__name__ not in ('BM4Midpoint', 'BM4Implicit', 'GaussLegendre4', 'RK2') or observers
			or options.record_step_statistics or options.progress):
		raise ValueError('Expected a supported method without observers, progress or step-statistic recording.')
	jax.config.update('jax_enable_x64', True)
	# A reused Modal container must not supply a warm first call. Persistent
	# compilation caching is disabled in the worker image as well.
	jax.config.update('jax_enable_compilation_cache', False)
	jax.clear_caches()
	_compiled_integrator.cache_clear()
	executor: Execution = Execution()
	solutions: dict[str, States] = {}
	measurements: list[dict[str, Any]] = []
	for repetition in range(repetitions + 1):
		order: tuple[str, str] = ('scipy', 'jax') if repetition % 2 == 0 else ('jax', 'scipy')
		for backend in order:
			choice: ExecutionOptions = replace(options, backend=backend)  # type: ignore[arg-type]
			before: Any = _compiled_integrator.cache_info()
			cpu_start: float = process_time()
			start: float = perf_counter()
			data: States = executor.run(problem, method, options=choice)
			seconds: float = perf_counter() - start
			cpu_seconds: float = process_time() - cpu_start
			after: Any = _compiled_integrator.cache_info()
			if backend == 'jax':
				if repetition == 0 and after.misses - before.misses != 1:
					raise RuntimeError('The first JAX call did not compile exactly one integration.')
				if repetition > 0 and (after.misses != before.misses or after.hits - before.hits != 1):
					raise RuntimeError('A warmed JAX call did not reuse the executable.')
			if (data.states.dtype != np.float64 or not np.isfinite(data.states).all()
					or data.diagnostics['output_interpolation_count'] != 0
					or data.diagnostics.get('record_step_statistics') is not False
					or any(isinstance(v, np.ndarray) and key not in ('tableau_A', 'tableau_B', 'tableau_C')
						for key, v in data.diagnostics.items())):
				raise ValueError('Expected finite float64 states, aligned samples and no diagnostic histories.')
			# Runge--Kutta coefficient arrays are fixed method metadata, not histories.
			if type(method).__name__ in ('GaussLegendre4', 'RK2'):
				tableau: Any = getattr(method, 'tableau')
				for letter in ('A', 'B', 'C'):
					np.testing.assert_array_equal(data.diagnostics[f'tableau_{letter}'],
						getattr(tableau, letter))
			if backend in solutions:
				np.testing.assert_array_equal(data.t, solutions[backend].t)
				np.testing.assert_array_equal(data.states, solutions[backend].states)
			else:
				solutions[backend] = data
			measurements.append(dict(backend=backend, repetition=repetition,
				phase='first' if repetition == 0 else 'warm', seconds=seconds,
				process_seconds=cpu_seconds, integration_cache_misses=after.misses-before.misses,
				integration_cache_hits=after.hits-before.hits))
			print(f'{type(method).__name__} {problem.request.t_span[1]:g} cycles '
				f'{backend} repetition {repetition}: {seconds:.6f} s', flush=True)
	np.testing.assert_array_equal(solutions['scipy'].t, solutions['jax'].t)
	# Coordinates are unwrapped and both computations start from identical
	# states. This discrepancy measures backend agreement, not solution error.
	delta: np.ndarray = solutions['scipy'].states - solutions['jax'].states
	n: int = problem.particle_count
	distance: np.ndarray = np.hypot(delta[:, :n], delta[:, n:])
	model: str = platform.processor()
	info: Path = Path('/proc/cpuinfo')
	if info.exists():
		model = next((line.split(':', 1)[1].strip() for line in info.read_text().splitlines()
			if line.startswith('model name')), model)
	environment: dict[str, Any] = dict(python=platform.python_version(), numpy=np.__version__,
		scipy=scipy.__version__, jax=jax.__version__, jaxlib=jaxlib.__version__,
		cpu_model=model, platform=platform.platform(), x64=bool(jax.config.read('jax_enable_x64')),
		visible_logical_cpus=os.cpu_count(),
		threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')})
	for name, path in (('cpu_max', '/sys/fs/cgroup/cpu.max'), ('memory_max', '/sys/fs/cgroup/memory.max')):
		environment[name] = Path(path).read_text().strip() if Path(path).exists() else 'unavailable'
	return dict(payload_sha256=sha256(payload).hexdigest(), solutions=solutions,
		measurements=measurements, environment=environment,
		agreement=dict(maximum_distance=float(distance.max()),
			rms_distance=float(np.sqrt(np.mean(distance**2))),
			final_maximum_distance=float(distance[-1].max()),
			maximum_distance_first_10_cycles=float(distance[:101].max()),
			convention='Euclidean distance in unwrapped base-cell coordinates; not an accuracy estimate'))


__all__ = ['benchmark_pair']
