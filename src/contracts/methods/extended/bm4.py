"""methods.extended.bm4 observation records."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from contracts.methods.observation import ImplicitIntegrationStep, IntegrationStage


@dataclass(frozen=True, slots=True)
class ImplicitBM4IntegrationStep(ImplicitIntegrationStep):
	"""Expose the converged projected-BM4 base cycle to exact observers."""

	coupling_frequency: float
	multiplier: np.ndarray = field(repr=False, compare=False)
	base_stages: tuple[IntegrationStage, ...] = field(repr=False, compare=False)


__all__ = ['ImplicitBM4IntegrationStep']
