# Proposed study

Evaluate the complete physical one-step maps of ABBA2Midpoint and BM4Midpoint, including duplication and arithmetic averaging, against their symmetric-projection controls ABBA2Implicit and BM4Implicit. At the 84 reference points and four forcing phases, use 20, 40, 80 and 160 fixed steps per cycle. Retain signed and absolute values of det(D Phi)-1 and the infinity-norm displacement after advancing by h and returning from t+h by -h in unwrapped base-cell coordinates.

Differentiate the actual map using the available analytic spatial Jacobians. For the arithmetic projections, compare the determinant defect with det(D e), where e is half the final difference between the two copies. Calibrate the implementation on a smooth Hamiltonian oscillator and distinguish roundoff or nonlinear-solver floors from resolved defects. Any observed power law is empirical; high formal orders require sufficient regularity and cannot be assigned to the reflected spline automatically.

At 40 steps per cycle, study accumulated log absolute determinants and forward/backward displacement after 10 and 100 cycles. Persist raw diagnostics, individual timings, parameters and provenance under the explicit prefix `studies/midpoint_geometry` and a run identifier. The final storage target is the configured scientific-results bucket; keep local recovery data and report publication failures accurately.

Provide calculation and visualization notebooks. Calculation is an explicit opt-in; the default workflow reuses saved results. Visualization loads those results and delegates thesis figures to the shared TFM plotting helper, with all figure text in French. The study and measured conclusions feed the rewritten title and content of thesis section 4.9.
