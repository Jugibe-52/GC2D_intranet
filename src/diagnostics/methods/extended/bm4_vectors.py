"""Capture planar projection and displacement vectors on accepted BM4 steps."""

import numpy as np
from contracts.observation import ImplicitBM4IntegrationStep, IntegrationStep


class BM4VectorObserver:
    """Retain only vectors and times, without keeping stage maps or closures."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.multipliers: list[np.ndarray] = []
        self.displacements: list[np.ndarray] = []

    def __call__(self, record: IntegrationStep) -> None:
        if not isinstance(record, ImplicitBM4IntegrationStep):
            raise TypeError("BM4 vector capture requires an implicit BM4 step.")
        # Physical packing is [x_1,...,x_p,y_1,...,y_p]. Store (particle, xy).
        self.times.append(record.time)
        self.multipliers.append(record.multiplier.reshape(2, -1).T.copy())
        self.displacements.append(
            (record.state_after - record.state_before).reshape(2, -1).T.copy()
        )
