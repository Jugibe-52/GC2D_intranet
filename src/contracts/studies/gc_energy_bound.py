"""studies.gc_energy_bound configuration and result records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


GC_ENERGY_BOUND_METHODS = ("BM4Implicit", "RK4")


@dataclass
class GCEnergyBoundResult:
	"""Every accepted node and independently auditable summary records."""

	arrays: dict[str, np.ndarray]
	metadata: dict[str, Any]
	summary: list[dict[str, Any]]
	envelopes: list[dict[str, Any]]
	blocks: list[dict[str, Any]]
	orders: list[dict[str, Any]]


__all__ = ['GC_ENERGY_BOUND_METHODS', 'GCEnergyBoundResult']
