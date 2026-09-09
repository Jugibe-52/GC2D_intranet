"""Planar BM4 vector histories with consistent particle and time alignment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from studies.bm4_vectors import BM4VectorStudy


def plot_bm4_vectors(study: BM4VectorStudy, labels: tuple[str, ...]) -> Figure:
    """Compare lengths, absolute directions, relative rotations and norm ratios."""
    if len(labels) != study.mu.shape[1]:
        raise ValueError("Provide one label per particle.")
    figure, axes = plt.subplots(4, len(labels), figsize=(14, 12), squeeze=False,
                                sharex=True, constrained_layout=True)
    for p, label in enumerate(labels):
        for values, name, color in (
            (study.mu_norm, r"$\|\mu_n\|_2$", "tab:purple"),
            (study.delta_norm, r"$\|z_{n+1}-z_n\|_2$", "tab:blue"),
        ):
            axes[0, p].semilogy(study.times, np.where(values[:, p] > 0, values[:, p], np.nan),
                                label=name, color=color)
        for values, name, color in (
            (study.mu_angle, r"$\arg\mu_n$", "tab:purple"),
            (study.delta_angle, r"$\arg\Delta z_n$", "tab:blue"),
        ):
            axes[1, p].plot(study.times, np.degrees(values[:, p]), label=name, color=color)
        axes[2, p].plot(study.times, np.degrees(study.angle_difference[:, p]), color="tab:green")
        axes[2, p].set(ylim=(-185, 185), yticks=(-180, -90, 0, 90, 180))
        axes[2, p].axhline(0, color="black", linewidth=.6)
        axes[3, p].plot(study.times, study.norm_ratio[:, p], color="tab:orange")
        axes[0, p].set_title(label)
        axes[0, p].legend()
        axes[1, p].legend()
        for row in range(4):
            axes[row, p].grid(alpha=.25)
            for boundary in range(1, int(study.times[-1]) + 1):
                axes[row, p].axvline(boundary, color="0.7", linewidth=.5)
        axes[3, p].set_xlabel(r"Step endpoint $t_{n+1}$ [normalized cycles]")
    for row, label in enumerate(("Euclidean norm", "Unwrapped direction [deg]",
                                  "Signed angle: delta to mu [deg]", "Norm ratio: mu / delta")):
        axes[row, 0].set_ylabel(label)
    return figure
