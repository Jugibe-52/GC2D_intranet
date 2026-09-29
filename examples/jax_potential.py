"""Compare SciPy and JAX potential evaluation on a reproducible particle batch."""

import argparse
from time import perf_counter
from collections.abc import Callable

import jax
import numpy as np

from contracts.execution_options import ExecutionOptions
from potential.potential import Potential
from potential.gc2d_h5 import load_gc2d_h5_potential


def _median_seconds(call: Callable[[], object], repeats: int) -> float:
    """Time completed calls; each JAX callable must synchronize its result."""
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        call()
        samples.append(perf_counter() - started)
    return float(np.median(samples))


def main() -> None:
    """Report compilation separately from warmed evaluation and transfers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--h5", help="Optional real HDF5 input; otherwise use a synthetic field.")
    args = parser.parse_args()
    if args.particles < 1 or args.repeats < 1:
        parser.error("Particle and repetition counts must be positive.")

    jax.config.update("jax_enable_x64", True)
    potential = (
        load_gc2d_h5_potential(args.h5)
        if args.h5 else Potential.random(A=0.7, M=8, nx=64, ny=64, seed=27, interpolation_order=3)
    ).gyroaverage(0.3)
    execution = ExecutionOptions(backend="jax", device=args.device)
    device = jax.devices(execution.device)[execution.device_index]
    rng = np.random.default_rng(27)
    x = potential.grid.xmin + potential.grid.period * rng.random(args.particles)
    y = potential.grid.ymin + potential.grid.period * rng.random(args.particles)
    time = 0.23
    td, xd, yd = jax.device_put((np.asarray(time), x, y), device)

    print(f"Field: {args.h5 or 'synthetic random potential'}; particles: {args.particles}")
    print(f"JAX {jax.__version__}; device: {device}; arithmetic: float64")
    for label, derivative in (("Potential", {}), ("Gradient x", {"dx": 1}),
                              ("Hessian xy", {"dx": 1, "dy": 1})):
        started = perf_counter()
        result = potential.evaluate(td, xd, yd, execution=execution, **derivative)
        result.block_until_ready()
        first_seconds = perf_counter() - started
        expected = potential.evaluate(time, x, y, **derivative)
        np.testing.assert_allclose(result, expected, rtol=3e-11, atol=3e-11)
        cpu = _median_seconds(lambda: potential.evaluate(time, x, y, **derivative), args.repeats)
        resident = _median_seconds(
            lambda: potential.evaluate(td, xd, yd, execution=execution, **derivative).block_until_ready(), args.repeats,
        )
        boundary = _median_seconds(
            lambda: np.asarray(potential.evaluate(time, x, y, execution=execution, **derivative)), args.repeats,
        )
        error = float(np.max(np.abs(np.asarray(result) - expected)))
        print(f"{label}: first call including preparation/compilation={first_seconds:.6f} s; max error={error:.3e}")
        print(f"  medians: SciPy={cpu*1e6:.1f} us; JAX resident={resident*1e6:.1f} us; "
              f"JAX with NumPy inputs/output={boundary*1e6:.1f} us")
    print("These are evaluator timings, not complete integration speedups.")


if __name__ == "__main__":
    main()
