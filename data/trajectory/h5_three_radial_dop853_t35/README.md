# Standard DOP853 reference through normalized time 35

Exact prefix copied from outputs/developements/accuracy/h5_three_radial_2x_precision/v1.
DOP853 states are the standard trajectory. Radau states and periodic distances
are retained solely for the independent numerical audit. No solver was rerun.

The interval is normalized time [0, 35], not 35 oscillation cycles, with 3501
samples spaced by 0.01. State layout: [x1, x2, x3, y1, y2, y3], shape (6, 3501).
Positions use the source characteristic-length normalization (0.06 m).
The source HDF5 potential is PHI_2.h5, B=1.5, indx=(0,1), cubic interpolation;
rho=0.3 and initial radial fractions=(0.1,0.2,0.3), angle=0.

Arithmetic: float64. DOP853 rtol=5e-13, atol=5e-15, maximum step=0.0025.
Radau audit rtol=5e-14, atol=5e-16, maximum step=0.00125.
Inherited solver work and runtime summaries describe the full source run to 200.

Maximum periodic DOP853–Radau discrepancy over this prefix: 5.504472094581e-06.
Per-particle maxima: [5.504472094580963e-06, 2.821845529359257e-07, 5.82193842740151e-08].
These are empirical discrepancies, not rigorous trajectory-error bounds.
No independent analytic solution is available for this interpolated measured field.

Load with diagnostics.load_reference_trajectory(directory); use .times and
.states for the DOP853 reference. The loader verifies the array checksum.
Keep trajectory.npz, metadata.json and README.md together. Use only with matching
physical settings, initial states and field fingerprint, within the saved interval.
Reproduce the export with export_standard_reference_t35.ipynb in the 2x notebook folder.
