"""Sample BM4 conjugate momentum and particle multipliers at output nodes."""
from typing import ClassVar
import numpy as np
from contracts.methods.extended.bm4 import ImplicitBM4IntegrationStep
from contracts.methods.observation import IntegrationStep
from dynamics import GuidingCenterDynamics
from diagnostics.methods.energy import EnergyObserver, PreparedEnergyObserver


class BM4EnergyBalanceObserver:
    """Reuse accepted-stage energy reconstruction independently for each particle.

    No time quadrature is performed on the sparse output grid. Multiplier block
    statistics include every accepted step, including unsaved intermediate peaks.
    """

    requires_quadrature: ClassVar[bool] = True

    def __init__(self, dynamics: GuidingCenterDynamics, particle_count: int, save_stride: int) -> None:
        self.dynamics = dynamics
        self.count = particle_count
        self.stride = save_stride
        self._energy: PreparedEnergyObserver | None = None
        self.momenta = [self.kappa.copy()]
        self.mu_endpoint = [np.zeros(particle_count)]
        self.mu_max = [np.zeros(particle_count)]
        self.mu_rms = [np.zeros(particle_count)]
        self.mu_mean = [np.zeros(particle_count)]
        self.global_statistics = [np.zeros(4)]
        self.block: list[np.ndarray] = []
        self.step_count = 0

    @property
    def kappa(self) -> np.ndarray:
        """Current physical momentum from the shared accepted-step accumulator."""
        return np.zeros(self.count) if self._energy is None else self._energy.momentum

    def __call__(self, record: IntegrationStep) -> None:
        """Accumulate exact BM4 stage increments and infinity norms per particle."""
        if not isinstance(record, ImplicitBM4IntegrationStep):
            raise TypeError("Expected a BM4 accepted-step record.")
        if record.step_index != self.step_count or record.dynamics is not self.dynamics:
            raise ValueError("BM4 records must be sequential and use the same dynamics.")
        if record.quadrature is None:
            raise ValueError("Energy observation requires accepted-stage quadrature.")
        if self._energy is None:
            self._energy = EnergyObserver().new_run(self.dynamics.hamiltonian,
                self.dynamics.extended_momentum_derivative,
                particle_count=self.count, initial_time=record.start_time)
        self._energy.accept(record.step_index, record.start_time, record.time, record.quadrature)
        n = self.count
        mu: np.ndarray = np.max(np.abs(record.multiplier.reshape(2, n)), axis=0)
        np.testing.assert_allclose(np.max(mu), record.projection_multiplier_norm)
        self.block.append(mu)
        self.step_count += 1
        if self.step_count % self.stride == 0:
            values: np.ndarray = np.asarray(self.block)
            self.momenta.append(self.kappa.copy())
            self.mu_endpoint.append(mu.copy())
            self.mu_max.append(values.max(axis=0))
            self.mu_rms.append(np.sqrt(np.mean(values**2, axis=0)))
            self.mu_mean.append(values.mean(axis=0))
            global_norm: np.ndarray = values.max(axis=1)
            self.global_statistics.append(np.array([global_norm[-1], global_norm.max(),
                                                    np.sqrt(np.mean(global_norm**2)), global_norm.mean()]))
            self.block.clear()
