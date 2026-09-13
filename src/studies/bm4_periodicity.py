"""Same-phase return and finite-horizon periodicity diagnostics for one BM4 orbit."""

import numpy as np


def periodicity_records(result, *, period):
    """Measure initial returns and shifted-trajectory agreement at integer lags.

    Each lag uses every saved step in the overlapping time interval. Distances
    use the minimum image on the spatial torus; time lags preserve forcing phase.
    """
    if result.config.particle_count != 1 or not np.isfinite(period) or period <= 0:
        raise ValueError('Require exactly one particle and a positive cell period.')
    samples = result.config.saved_samples_per_cycle
    states = result.positions[0]  # (2, samples), dimensionless x and y.
    delta = (states - states[:, :1] + period / 2) % period - period / 2
    distances = np.linalg.norm(delta, axis=0)
    returns, lags = [], []
    for cycle in range(1, result.config.cycle_count + 1):
        index = cycle * samples
        returns.append(dict(cycle=cycle, distance=float(distances[index]),
                            fraction=float(distances[index] / period),
                            dx=float(delta[0, index]), dy=float(delta[1, index])))
        if index < states.shape[1] - 1:
            difference = (states[:, index:] - states[:, :-index] + period / 2) % period - period / 2
            errors = np.linalg.norm(difference, axis=0)
            lags.append(dict(cycle=cycle, overlap_cycles=result.config.cycle_count - cycle,
                             pairs=int(errors.size), rms=float(np.sqrt(np.mean(errors**2))),
                             maximum=float(np.max(errors)), maximum_fraction=float(np.max(errors) / period)))
    return returns, lags
