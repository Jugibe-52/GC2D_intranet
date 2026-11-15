# Spatial extension and reflected Hamiltonians

Potential, Potential.load, Potential.random and the inherited JaxPotential
factories accept spatial_extension="reflected" (default) or "periodic".

```python
reflected = Potential.load("data/potential/V1/PHI_2.h5")
periodic = Potential.load("data/potential/V1/PHI_2.h5", spatial_extension="periodic")
```

The periodic option preserves the existing wrapped interpolation and FFT
gyroaverage. The reflected option uses cubic splines with zero normal
derivatives on the base square. It requires interpolation_order=3; periodic
interpolation retains degrees 2 through 5. Existing calls asking for a higher
degree must explicitly select periodic. Both classes use the same physical data
and prepared coefficient representation; class choice selects NumPy or JAX.

## Coordinates and regularity

Grid retains the half-open base samples and width L=n*dx. Its period attribute
still describes this sampling width. Potential.spatial_period is the physical
repeat length: 2L for reflected and L for periodic. Normalization by L and the
physical meaning of rho do not change.

For each coordinate, u=(x-x0) mod 2L is folded to x0+u when u<=L, otherwise
x0+2L-u. The local derivative signs sx,sy multiply a scalar derivative of order
(a,b) by sx**a*sy**b. The mean and each complex temporal coefficient use tensor
cubic interpolation with zero first derivative at both ends of each axis.
These conditions hold along whole edges; mixed derivatives vanish where
required by differentiating the edge conditions.

The even extension is C2 in space, including seams and corners. GC uses the
unchanged canonical field f=(-H_y,H_x), H=Phi_rho/B, which is C1 in space.
Its tangential component vanishes on a base edge; its normal component may
cross it. Both components vanish at a corner. No particle reflection, boundary
termination or alternating canonical rotation is applied by the integrator.
Finite Fourier time dependence supplies smooth temporal coefficients and
bounded temporal factors. Cubic C2 regularity alone does not justify arbitrary
higher-order smoothness assumptions for numerical convergence proofs.

CPU evaluation uses public SciPy tensor splines. JAX evaluates the identical
knots and coefficients with analytic differentiated B-spline bases. At an exact
seam choose the same one-sided +/-1 sign for coordinates and derivatives.
Do not mask first derivatives to zero: that would erase their nonzero second
derivatives under automatic differentiation.

## Missing endpoints and gyroaveraging

Half-open source data omit the upper edges. On initial construction they are
completed with their previous periodic trace, without moving or changing source
samples. This reproducible completion is independent of the zero-normal
boundary condition. In particular, reflected models do not require opposite
edges to have equal values.

The optional reflected_samples constructor argument supplies all closed samples,
with shape (1+mode_count,nx+1,ny+1), leading real mean and subsequent complex
modes. Its interior must match the half-open mean/modes within assembly roundoff;
the canonical half-open samples are retained exactly. The property is read-only
and is None for periodic potentials.

For nonzero rho, reflecting the closed samples produces a 2nx by 2ny half-open
FFT block. Its angular wave numbers define J0(rho*|k|). After inverse FFT the
entire closed base block is retained and clamped splines are reconstructed.
Upper edges are independent after averaging and must never be rebuilt by
copying lower edges. This is a spectral average of the sampled field followed
by spline reconstruction, not exact circular quadrature of a cubic spline.
rho=0 returns the same potential. Repeated averaging, magnetic scaling, pickles,
worker snapshots and saved solutions preserve extension and closed samples.
Saved reflected solutions use archive schema 3, which requires explicit model
metadata and closed samples; older readers reject this version. Periodic
solutions and archives without a potential retain schema 2. The current reader
also accepts historical schemas 1 and 2 with their periodic interpretation.

## Display and saved studies

Integration and States remain unwrapped. display_coordinates folds copies into
the base square; display_vectors transforms a particle's actual velocity using
the signs at its original physical coordinates. The resulting display path can
turn at a fold even though its physical trajectory crosses smoothly.

In reflected GC, a folded branch of reversed orientation has velocity opposite
to the canonical base-tile arrow at the same displayed location. Background
quivers therefore identify the base tile; particle arrows use their actual
branch. Physical minimum-image distances use spatial_period, never the folded
display coordinates.

New configurations and archives record the selected extension. Old untagged
pickles, snapshots and processed fields restore periodic explicitly. Historical
study reconstruction retains its old periodic model and fingerprint. Newly
configured studies use reflected unless periodic is selected. Poincare
automation exposes the option in AutomationConfig and the --spatial-extension
CLI argument; its legacy configuration reader preserves original keys and
digests. See [automation](../studies/poincare-automation.md).

Implementation sources: [Potential](../../src/potential/potential.py),
[spatial transforms](../../src/potential/_spatial.py), and
[constrained splines](../../src/potential/_reflected_spline.py).
SciPy documents [clamped boundary conditions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html)
and [tensor B-spline evaluation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdBSpline.html).
