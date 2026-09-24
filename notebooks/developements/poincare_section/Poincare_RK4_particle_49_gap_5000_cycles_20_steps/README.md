# RK4 particle 49 in the sampled gap

1. Run `calculo.ipynb` to calculate or reload particle 49.
2. Run `visualizacion.ipynb` to create `poincare_rk4_particle49.html` and its local URL.

The new particle starts at (x/L, y/L) = (0.4400, 0.5805), with color #d000d0.
Defaults: classical RK4, rho = 0.3, t0 = 0, 5000 unit forcing cycles, 20 steps per
cycle, 500-step integration blocks, and every step saved in float64. The local URL
is available while the visualization kernel runs. The standalone HTML also opens
directly in a browser and provides cycle-range and particle-selection controls.

The 48 original RK4 particles are loaded from the sibling study
`Poincare_RK4_48_radiales_5000_ciclos_20_steps_local`, run
`local_rk4_48p_5000c_20s_20260920`, with verified checksums. Their files, IDs and
colors remain intact. The GC particles are uncoupled, so only particle 49 is newly
integrated. Other methods are outside this focused seeding experiment.

`assets/gc2d_snapshot.zip` is the original 1,628,508-byte field/source archive,
SHA-256 eae45bb3f0c1ff0ee7d27985eaee6fe753dfc1cc2353e1ae75d3a66e09ed4e0d.
Only its normalized field arrays and provenance are loaded; the calculation uses
the current public GC/RK4 APIs and records their source hashes. Settings retain
B = 1.5 T, characteristic length = 0.06 m, selection (0, 1), cubic interpolation,
and the original forcing phase and normalization.

Results are stored under `results/rk4_particle49_5000c_20s/`: trajectory.npz,
metadata.json, COMPLETE.json and particle49_gap.png. Choose another result
directory if changing inputs. Complete runs are never silently overwritten.

The seed was selected by maximal minimum periodic distance to the 240,000 original
cycle returns on a 251 × 391 grid over x/L [0.425, 0.450] and y/L [0.557, 0.596].
The initial clearance is about 0.01164235 L. This concerns the sampled record, not
a proof of inaccessible phase space, exact periodicity, or trajectory accuracy.
The viewer renders all saved returns in float32; the trajectory remains float64.

These development notebooks and their outputs remain local and Git-ignored.
