"""Plot saved worker-level Newton and projection-multiplier diagnostics."""
import numpy as np
import matplotlib.pyplot as plt


def plot_diagnostics(metadata, arrays, output, max_blocks=1000):
    """Display block maxima without discarding the complete saved histories."""
    n = metadata['complete_steps']
    block_size = max(1, int(np.ceil(n / max_blocks)))
    starts = np.arange(0, n, block_size)
    ends = np.minimum(starts + block_size, n)
    cycles = arrays['times'][ends] / metadata['cycle_duration']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    ratio = arrays['nonlinear_residuals'] / arrays['nonlinear_tolerances']
    series = [arrays['nonlinear_iterations'], ratio, arrays['projection_multiplier_norms']]
    for ax, values, title in zip(axes.flat, series, [
        'Newton corrections', 'Final residual / tolerance', 'Projection multiplier infinity norm',
    ]):
        for w in range(metadata['process_count']):
            ax.plot(cycles, np.maximum.reduceat(values[w], starts), lw=0.8, label=f'Worker {w + 1}')
        ax.set(xlabel='Forcing cycles', ylabel=title,
               title=f'{title}: maxima over {block_size} steps')
        ax.grid(alpha=0.2)
    axes[0, 1].axhline(1, color='black', ls='--', lw=1)
    axes[1, 0].set_yscale('symlog', linthresh=1e-15)
    ax = axes[1, 1]
    for w in range(metadata['process_count']):
        # Select a recorded solve with the highest correction count per worker.
        step = int(np.argmax(arrays['nonlinear_iterations'][w]))
        lo, hi = arrays['newton_history_offsets'][w, step:step + 2]
        values = arrays['newton_history_residuals'][lo:hi]
        threshold = arrays['nonlinear_tolerances'][w, step]
        ax.plot(np.arange(len(values)), values / threshold, '.-',
                label=f'Worker {w + 1}, step {step + 1}')
    ax.set(xlabel='Newton iteration (0 = initial guess)', ylabel='Residual / tolerance',
           title='Recorded solves with maximum Newton corrections')
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.axhline(1, color='black', ls='--', lw=1)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'BM4 Newton and mu diagnostics | {metadata["cycles"]} cycles | '
                 f'{metadata["steps_per_cycle"]} steps/cycle')
    fig.savefig(output / 'newton_mu_diagnostics.png')
    fig.savefig(output / 'newton_mu_diagnostics.svg')
    return fig
