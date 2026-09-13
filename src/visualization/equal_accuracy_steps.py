"""Equivalent-step ratios across resolved trajectory-error targets."""

from collections.abc import Mapping, Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from studies.equal_accuracy_steps import EqualAccuracyStep


def plot_equal_accuracy_steps(rows_by_variant: Mapping[str, Sequence[EqualAccuracyStep]]) -> tuple[Figure, Axes]:
    """Plot h_ABBA4/h_BM4 without filling gaps in unsupported targets."""
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, rows in rows_by_variant.items():
        if rows:
            axis.semilogx([r.target_error for r in rows], [r.step_ratio for r in rows],
                          "o", label=label)
    axis.axhline(1.0, color="black", linestyle="--", label="Equal steps")
    axis.set(xlabel="Target space-time RMS trajectory error", ylabel="h_ABBA4 / h_BM4",
             title="Interpolated step ratio at equal accuracy")
    axis.grid(alpha=0.3)
    axis.legend()
    return figure, axis
