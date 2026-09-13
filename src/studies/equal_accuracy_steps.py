"""Infer equivalent steps from resolved, monotone accuracy refinements."""

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np


@dataclass(frozen=True)
class EqualAccuracyStep:
    """Interpolated steps at one shared trajectory-error target."""

    target_error: float
    bm4_step: float
    abba4_step: float
    step_ratio: float
    bm4_local_order: float
    abba4_local_order: float


def _segments(steps: Sequence[float], errors: Sequence[float], floor: float) -> list[tuple[float, float, float, float]]:
    """Keep adjacent increasing error brackets above the reference threshold."""
    h, e = np.asarray(steps, dtype=float), np.asarray(errors, dtype=float)
    if h.ndim != 1 or h.size < 2 or h.shape != e.shape:
        raise ValueError("Steps and errors must be matching vectors with at least two entries.")
    if not np.all(np.isfinite(h)) or np.any(h <= 0) or np.unique(h).size != h.size:
        raise ValueError("Steps must be finite, positive and unique.")
    indices = np.argsort(h)
    h, e = h[indices], e[indices]
    # Do not bridge a failed, unresolved, or nonmonotone refinement point.
    return [(float(h0), float(h1), float(e0), float(e1))
            for h0, h1, e0, e1 in zip(h[:-1], h[1:], e[:-1], e[1:])
            if np.isfinite(e0) and np.isfinite(e1) and floor < e0 < e1]


def _inverse(segments: list[tuple[float, float, float, float]], target: float) -> tuple[float, float] | None:
    """Invert one unambiguous log-log bracket without extrapolating."""
    candidates = []
    for h0, h1, e0, e1 in segments:
        if e0 <= target <= e1:
            order = float(np.log(e1 / e0) / np.log(h1 / h0))
            step = float(h0 * np.exp(np.log(target / e0) / order))
            candidates.append((step, order))
    if not candidates:
        return None
    # Shared endpoints are harmless; distinct branches are ambiguous.
    if not np.allclose([c[0] for c in candidates], candidates[0][0], rtol=1e-10, atol=0):
        return None
    return candidates[0]


def equal_accuracy_steps(
    steps: Sequence[float], bm4_errors: Sequence[float], abba4_errors: Sequence[float],
    *, reference_floor: float, floor_factor: float = 10.0,
    targets: Sequence[float] | None = None, target_count: int = 9,
) -> tuple[EqualAccuracyStep, ...]:
    """Return h_ABBA4/h_BM4 at common errors; omit unsupported targets.

    Log-log interpolation assumes a local power law, not order four. The
    reference discrepancy is an empirical resolution estimate, not an error
    bound. Empty output means the sweep cannot resolve an equivalent step.
    """
    if not np.isfinite(reference_floor) or reference_floor < 0:
        raise ValueError("Reference floor must be finite and nonnegative.")
    if not np.isfinite(floor_factor) or floor_factor < 1:
        raise ValueError("Floor factor must be finite and at least one.")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 2:
        raise ValueError("Target count must be an integer of at least two.")
    floor = max(float(reference_floor * floor_factor), float(np.finfo(float).tiny))
    bm = _segments(steps, bm4_errors, floor)
    ab = _segments(steps, abba4_errors, floor)
    if targets is not None:
        target_values = np.asarray(targets, dtype=float)
        if target_values.ndim != 1 or not np.all(np.isfinite(target_values)) or np.any(target_values <= 0):
            raise ValueError("Targets must be a vector of finite positive errors.")
    elif bm and ab:
        low = max(min(s[2] for s in bm), min(s[2] for s in ab))
        high = min(max(s[3] for s in bm), max(s[3] for s in ab))
        if low > high:
            return ()
        target_values = np.geomspace(low, high, target_count) if low < high else np.array([low])
    else:
        return ()
    rows = []
    for target in target_values:
        b, a = _inverse(bm, float(target)), _inverse(ab, float(target))
        if b is not None and a is not None:
            rows.append(EqualAccuracyStep(float(target), b[0], a[0], a[0] / b[0], b[1], a[1]))
    return tuple(rows)
