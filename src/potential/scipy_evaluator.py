"""SciPy execution of the common prepared-potential evaluation contract."""

from typing import Any

import numpy as np

from ._evaluation import PotentialEvaluator, normalize_coordinates, reconstruct


class ScipyPotentialEvaluator(PotentialEvaluator):
    """Evaluate the canonical SciPy interpolants with host NumPy arrays."""

    namespace = np

    def asarray(self, value: Any) -> np.ndarray:
        """Retain the established NumPy conversion convention."""
        return np.asarray(value)

    def _evaluate(self, time: np.ndarray, x: np.ndarray, y: np.ndarray,
                  dx: int, dy: int, dt: int) -> Any:
        """Interpolate in space, then apply the shared harmonic reconstruction."""
        x, y = normalize_coordinates(x, y, (self.grid.xmin, self.grid.ymin), self.grid.period)
        fields = self.prepared.spatial_fields(x, y, dx, dy)
        return reconstruct(time, fields, self.prepared.frequencies, dt, xp=np)

    def _evaluate_grid(self, time: np.ndarray, dt: int) -> Any:
        """Reconstruct the stored samples without spatial interpolation."""
        fields = self.prepared.samples
        fields = fields.reshape(fields.shape + (1,) * time.ndim)
        return reconstruct(time, fields, self.prepared.frequencies, dt, xp=np)


__all__ = ["ScipyPotentialEvaluator"]
