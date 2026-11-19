# Third-set Poincare section

Open `notebooks/third_set_poincare.ipynb` in the existing GC2D Python environment and run its cells. The notebook computes the eight particles in `data/particles/chaotic_candidates.json`, preserving their IDs, exact physical positions and colors. It does not reuse the older eight-particle historical selection.

The default calculation uses BM4Implicit, rho = 0.30, B = 1.5 T, 5000 physical forcing cycles and 40 fixed steps per cycle. The measured potential uses constant field 0 and variable field 15, cubic interpolation, no smoothing and the reflected spatial extension. Both loader rescalings are enabled. The physical gyro-radius is rho*0.06/(2*pi) metres. Newton uses analytic Jacobians, absolute tolerance 1e-12, relative tolerance 1e-11 and at most 40 corrections. Execution uses local JAX CPU float64.

The first parameter cell controls rho, cycle count, fixed steps per cycle, magnetic field, archive reuse, export and plot settings. `recompute = False` only reuses archives matching all input, potential, code and numerical-library fingerprints. `export_thesis_figure = False` disables copying the requested figure to the sibling thesis directory.

Outputs live in `data/trajectory/third_set_poincare/`: compressed NPZ returns, JSON metadata and French-labelled PNG/PDF figures. Whole-cycle output contains 5001 time rows and 16 state components for the default eight-particle run. The initial row is saved but excluded from the plotted cloud. The figure displays cycles 1–5000 by default, using `first_cycle = 1` and `last_cycle = 5000`; changing this interval does not repeat the integration. The notebook checks finite float64 values, unchanged initial coordinates, exact output times, 200000 fixed steps and zero output interpolation. Nonlinear failures are rejected by the existing integrator.

Coordinates remain unwrapped. The default unit-square viewport matches the supplied visual example; it does not impose a boundary on the dynamics. Points outside that viewport are counted and retained in the archive. Change `view_limits` and rerun the plotting cell to inspect them. The plot retains one original color per particle; `show_legend` enables a French particle legend.

Thesis exports are `TFM_redaccion/figuras/tercer_conjunto_poincare_fr.png`, the corresponding PDF and JSON provenance. The plot alone does not verify chaos, confinement or convergence at long times. No solver or potential implementation is modified. See the [main reflected campaign](main-reflected-campaign.md) for the source settings.

## Recorded default result

The 2026-11-16 execution completed 200000 fixed steps in 1378.57 seconds including JAX compilation. Its archive is `data/trajectory/third_set_poincare/bm4implicit_2e069839791cac95.npz`. Of 40000 non-initial returns, 1897 are outside the default viewport; their unwrapped coordinates remain saved. These counts describe the complete 5000-cycle archive and the interval selected in the current thesis figure.
