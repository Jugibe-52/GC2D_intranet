"""Synchronized JAX timings of eight fixed-step methods on one CPU worker."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import platform
from time import perf_counter, process_time
from typing import Any

import numpy as np

from execution._modal_worker import decode_job
from execution.execution import Execution
from states.history import States

METHODS = ('RK2', 'ImplicitMidpoint', 'RK4', 'GaussLegendre4',
	'ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit')


def benchmark_methods(payloads: dict[str, bytes], repetitions: int = 3) -> dict[str, Any]:
	"""Measure first calls and cached repetitions, without trajectory distances.

	All methods run sequentially in this process. Returning host-owned States
	synchronizes JAX before stopping the clock. Audits occur outside timing.
	"""
	if set(payloads) != set(METHODS) or type(repetitions) is not int or repetitions < 3:
		raise ValueError('Supply all eight methods and at least three cached repetitions.')
	import jax
	import jaxlib
	import scipy
	from execution._jax_fixed import _compiled_integrator

	jobs: dict[str, Any] = {name: decode_job(payloads[name]) for name in METHODS}
	first_problem: Any = jobs[METHODS[0]][0]
	for name, (problem, method, options, observers) in jobs.items():
		if (type(method).__name__ != name or options.backend != 'jax' or options.device != 'cpu'
				or observers or options.progress or options.record_step_statistics
				or problem.particle_count != 40):
			raise ValueError('Expected forty particles, JAX CPU and no optional diagnostics.')
		np.testing.assert_array_equal(problem.initial_state, first_problem.initial_state)
		np.testing.assert_array_equal(problem.request.output_times, first_problem.request.output_times)
	jax.config.update('jax_enable_x64', True)
	jax.config.update('jax_enable_compilation_cache', False)
	jax.clear_caches()
	_compiled_integrator.cache_clear()
	executor: Execution = Execution()
	solutions: dict[str, States] = {}
	measurements: list[dict[str, Any]] = []
	started: str = datetime.now(timezone.utc).isoformat()
	for repetition in range(repetitions + 1):
		order: tuple[str, ...] = METHODS if repetition % 2 == 0 else METHODS[::-1]
		for position, name in enumerate(order):
			problem, method, options, _ = jobs[name]
			before: Any = _compiled_integrator.cache_info()
			cpu_start: float = process_time()
			start: float = perf_counter()
			data: States = executor.run(problem, method, options=options)
			seconds: float = perf_counter() - start
			cpu_seconds: float = process_time() - cpu_start
			after: Any = _compiled_integrator.cache_info()
			misses: int = after.misses - before.misses
			hits: int = after.hits - before.hits
			if (misses, hits) != ((1, 0) if repetition == 0 else (0, 1)):
				raise RuntimeError(f'Unexpected integration compilation cache state for {name}.')
			if (not data.complete or data.states.dtype != np.float64
					or data.states.shape != (len(problem.request.output_times), 80)
					or not np.isfinite(data.states).all()
					or data.diagnostics['execution_backend'] != 'jax'
					or data.diagnostics['output_interpolation_count'] != 0
					or data.diagnostics.get('record_step_statistics') is not False
					or any(isinstance(v, np.ndarray) and key not in ('tableau_A', 'tableau_B', 'tableau_C')
						for key, v in data.diagnostics.items())):
				raise ValueError(f'Incomplete result or unexpected diagnostics for {name}.')
			np.testing.assert_array_equal(data.t, problem.request.output_times)
			np.testing.assert_array_equal(data.states[0], problem.initial_state)
			if name in solutions:
				# A determinism audit is not a particle-distance statistic.
				np.testing.assert_array_equal(data.states, solutions[name].states)
			else:
				solutions[name] = data
			measurements.append(dict(method=name, repetition=repetition, position=position,
				phase='first' if repetition == 0 else 'warm', seconds=seconds,
				process_seconds=cpu_seconds, integration_cache_misses=misses,
				integration_cache_hits=hits, step_count=int(data.diagnostics['step_count'])))
			print(f'{name} repetition {repetition}: {seconds:.6f} s', flush=True)
	model: str = platform.processor() or 'unavailable'
	info: Path = Path('/proc/cpuinfo')
	if info.exists():
		model = next((line.split(':', 1)[1].strip() for line in info.read_text().splitlines()
			if line.startswith('model name')), model)
	environment: dict[str, Any] = dict(python=platform.python_version(), numpy=np.__version__,
		scipy=scipy.__version__, jax=jax.__version__, jaxlib=jaxlib.__version__,
		cpu_model=model, platform=platform.platform(), x64=bool(jax.config.read('jax_enable_x64')),
		visible_logical_cpus=os.cpu_count(), devices=[str(d) for d in jax.devices()],
		threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')})
	return dict(payload_sha256={name: sha256(value).hexdigest() for name, value in payloads.items()},
		solutions=solutions, measurements=measurements, environment=environment,
		started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
		same_container=True, repeated_states_identical=True)


__all__ = ['METHODS', 'benchmark_methods']
