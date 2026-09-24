# Two sampled gaps with RK4 and BM4Midpoint

Run `calculo.ipynb` to calculate or reload the probe batches, then
`visualizacion.ipynb` to export the self-contained HTML and start its local URL.

| Particle | Initial x/L | Initial y/L | Color | Region |
|---|---:|---:|---|---|
| 49 | 0.4400 | 0.5805 | #d000d0 | Original gap |
| 50 | 0.6080 | 0.5663 | #a000d4 | First new image |
| 51 | 0.4321 | 0.3849 | #111111 | Second new image |

Both methods use 5000 unit forcing cycles, 20 complete steps per cycle, rho=0.3,
t0=0, and identical initial positions. BM4Midpoint uses coupling pi/8, twelve
stages per complete step and arithmetic-mean projection, with no Newton solve.
Its copy-separation diagnostic covers the infinity norm of the whole probe batch,
not individual particles or an implicit multiplier. All 100001 states are saved
in float64 and all 5000 cycle returns are displayed. No accuracy reference is run.

Seeds 50 and 51 maximize minimum periodic distance to the union of the original
RK4 and BM4Midpoint saved returns on grids explicitly specified in the calculation
notebook and `assets/selected_regions.json`. This is a finite-record gap search,
not proof of inaccessible phase space or area-filling behavior.

The 48 original trajectories are checksum verified and reused from their sibling
RK4 and BM4Midpoint studies. Particle 49's completed RK4 run is reused from
`../Poincare_RK4_particle_49_gap_5000_cycles_20_steps/results/rk4_particle49_5000c_20s`.
RK4 newly integrates particles 50 and 51; BM4Midpoint newly integrates 49, 50 and 51.
These GC particles do not interact. The two batch runtimes are not a method benchmark.

The original field/source archive is copied locally to `assets/gc2d_snapshot.zip`:
SHA-256 eae45bb3f0c1ff0ee7d27985eaee6fe753dfc1cc2353e1ae75d3a66e09ed4e0d.
Only its field arrays and provenance are loaded. Current public numerical APIs
are used and their hashes recorded. Physical settings preserve B=1.5 T,
characteristic length 0.06 m, selection (0,1), cubic interpolation and the original
unit forcing period. RK4 keeps its original chunk clock; BM4Midpoint keeps the
original global clock t0+k*h.

Results live in `results/rk4_5000c_20s/` and `results/bm4midpoint_5000c_20s/` as
trajectory.npz, metadata.json and checksum completion manifests. Choose a new
directory when changing settings. Complete results are not overwritten.

The HTML `poincare_two_regions_rk4_bm4midpoint.html` has one panel per method,
synchronized region selection, cycle controls, zoom/pan and particle selection.
Particle 50 is violet and 51 black for contrast. The earlier magenta particle 49
is retained. HTML display uses float32 only; saved calculations remain float64.
The local URL lasts while the visualization kernel runs; the HTML is also portable.

Development notebooks and their products remain local and Git-ignored.
