# BM4Implicit Poincare experiment

Independent reproducible study with 24 original particles, 5,000 cycles, 40 complete steps per cycle and 16 processes. Particle IDs and colors are inherited from `assets/original_particles.json`; IDs are never renumbered. No reference is computed.

Run `python run.py calculate --run-id aws_bm4implicit_24odd_5000c_40s_20260920 --resume --timeout 28800`, then `python run.py visualize --run-id aws_bm4implicit_24odd_5000c_40s_20260920`. Use `--smoke` with a new run ID for two-cycle validation. Numerical outputs are in `resultados/<run-id>` and figures and the offline particle selector in `figuras/<run-id>`.

The checked-in reusable runtime source is under `src/studies/poincare_batch_runtime` and plotting source under `src/visualization`. The local files here are deployment copies frozen and checksummed with the input package. The original scientific source snapshot is unmodified. Checkpoints commit every 1000 steps; restart uses original global times. For implicit runs, all Newton residuals and multiplier infinity norms per group are saved. Midpoint runs record copy separation and have no Newton or implicit multiplier records.

AWS uses one c8a.4xlarge Spot instance, a $0.50/hour maximum price, encrypted storage and a 8-hour global deadline including interruptions. Results and checkpoints are uploaded to private S3; the instance stops after successful export or failure.

## Periodicity of particles 1, 3, 5 and 7

Open `visualizacion_periodicidad_1_3_5_7.ipynb` in the project Python environment. It analyzes short recurring rhythms in the checksum-verified saved cycle CSV without integrating trajectories or requiring the full trajectory NPZ. The first qualifying autocorrelation peak within 20 cycles defines a candidate; later peaks through 80 cycles confirm its cadence. Complete geometric revolutions and disjoint 500-cycle windows quantify timing stability. Multiple-shift defects, phase classes and a first-half-trained, second-half-evaluated template test whether a fixed integer-cycle pattern persists. The particle-5 eight-cycle structure is shown separately. Long-lag minima do not select the rhythm.

Current figures are saved under `figuras/<run-id>/short_periodicity_particles_1_3_5_7/`; diagnostic tables, computed interpretation and source provenance are under `resultados/<run-id>/short_periodicity_particles_1_3_5_7/`. Earlier long-recurrence outputs remain in `periodicity_particles_1_3_5_7/`. All diagnostics concern once-per-cycle numerical samples; fractional revolution times use angular interpolation. Geometric thresholds do not certify exact physical periodicity or within-cycle repetition. Reusable analysis, plotting and export helpers live in `src/studies/poincare_periodicity.py`, `src/visualization/poincare_periodicity.py` and `src/diagnostics/poincare_periodicity.py`, respectively.

## Long recurrences of particles 1 and 3

Open `long_periodicity_particles_1_3.ipynb` to analyze only the original particle IDs 1 and 3 using the same verified saved data. The default search tests every integer lag from 1 to 1,000 forcing cycles and requires at least five candidate-period lengths in the full record. It ranks RMS repetition defects, reports maximum and 95th-percentile defects, compares both halves of the record, and shows the complete best-lag defect histories, initial returns and spectral context. Search controls and geometric thresholds are editable; conclusions are computed from the data.

No integration is performed. Figures and diagnostic exports use separate `long_periodicity_particles_1_3/` directories under `figuras/<run-id>/` and `resultados/<run-id>/`. The self-contained `interactive_space_and_initial_returns.html` viewer stacks particles 1 and 3 vertically. Its shared inclusive start/end cycle selection filters both the spatial returns and the initial-state-distance graphs. Long approximate recurrences are distinguished from the short spectral timescale and from exact physical periodicity.
