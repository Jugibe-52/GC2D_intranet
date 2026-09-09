"""Sample BM4 conjugate momentum and particle multipliers at output nodes."""
from dataclasses import replace
import numpy as np
from simulation import ImplicitBM4IntegrationStep
from .energy.observer import _bm4_stage_kappa_increment


class BM4EnergyBalanceObserver:
    """Reuse accepted-stage energy reconstruction independently for each particle.

    No time quadrature is performed on the sparse output grid. Multiplier block
    statistics include every accepted step, including unsaved intermediate peaks.
    """

    def __init__(self, dynamics, particle_count, save_stride):
        self.dynamics = dynamics
        self.count = particle_count
        self.stride = save_stride
        self.kappa = np.zeros(particle_count)
        self.momenta = [self.kappa.copy()]
        self.mu_endpoint = [np.zeros(particle_count)]
        self.mu_max = [np.zeros(particle_count)]
        self.mu_rms = [np.zeros(particle_count)]
        self.mu_mean = [np.zeros(particle_count)]
        self.global_statistics = [np.zeros(4)]
        self.block = []
        self.step_count = 0

    def __call__(self, record):
        """Accumulate exact BM4 stage increments and infinity norms per particle."""
        if not isinstance(record, ImplicitBM4IntegrationStep):
            raise TypeError("Expected a BM4 accepted-step record.")
        if record.step_index != self.step_count or record.dynamics is not self.dynamics:
            raise ValueError("BM4 records must be sequential and use the same dynamics.")
        n = self.count
        for p in range(n):
            indices = [p, n + p, 2*n + p, 3*n + p]
            for stage in record.base_stages:
                # The scalar reconstruction reads only stage snapshots and times.
                scalar_stage = replace(stage, state_before=stage.state_before[indices],
                                       state_after=stage.state_after[indices])
                self.kappa[p] += _bm4_stage_kappa_increment(
                    scalar_stage, self.dynamics, coupling_frequency=record.coupling_frequency)
        mu = np.max(np.abs(record.multiplier.reshape(2, n)), axis=0)
        np.testing.assert_allclose(np.max(mu), record.projection_multiplier_norm)
        self.block.append(mu)
        self.step_count += 1
        if self.step_count % self.stride == 0:
            values = np.asarray(self.block)
            self.momenta.append(self.kappa.copy())
            self.mu_endpoint.append(mu.copy())
            self.mu_max.append(values.max(axis=0))
            self.mu_rms.append(np.sqrt(np.mean(values**2, axis=0)))
            self.mu_mean.append(values.mean(axis=0))
            global_norm = values.max(axis=1)
            self.global_statistics.append(np.array([global_norm[-1], global_norm.max(),
                                                    np.sqrt(np.mean(global_norm**2)), global_norm.mean()]))
            self.block.clear()
