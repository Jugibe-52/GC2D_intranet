# Proposed representation and boundary decision

For a base square with origin `(x0, y0)` and width `L`, fold each axis using
`u = (x - x0) % (2*L)` and `x_m = x0 + L - abs(u - L)`.
Within each open cell, retain the orientation signs `sx, sy` in addition to the
folded coordinates. Scalar derivative order `(a, b)` receives `sx**a * sy**b`;
displayed velocities receive `(sx, sy)` componentwise. Saved positions remain
unchanged. These rules do not define a derivative at the folding corners.

## Decision required before implementation

With reflected Hamiltonian `H_m`, fixed rotation gives `v = (-H_m,y, H_m,x)`.
The previously discussed geometrically reflected velocity instead uses
`v = sx*sy * (-H_m,y, H_m,x)` within each cell. Its normal component changes
sign at a shared edge. For example, `H = cos(2*pi*y)` gives velocity `(2*pi, 0)`
at `y=1/4` in the original cell and `(-2*pi, 0)` in its right-hand mirror.
Both trajectories approach their shared edge; ordinary continuous-state
solutions cannot cross it transversally. Smooth scalar interpolation and
gyroaveraging do not resolve this problem.

The user must select one of these different behaviours:

- Reflect the potential, keep the original rotation, and construct a smooth
  Hamiltonian field across the edges. Velocities follow the scalar derivatives;
  they are not the geometric mirror of the original arrows.
- Preserve the existing physical dynamics and apply reflection only for
  display. Transform each particle's actual velocity together with its position;
  a single background field need not describe every folded branch.
- Reflect both potential and velocity, alternate the rotation within cells,
  and terminate a run at its first boundary contact. This requires explicit
  boundary detection and does not claim a globally crossing symplectic flow.

No boundary behaviour is selected by this document. A different requested
crossing law will require its own mathematical definition and impact review.

## Potential and gyroaverage if physical reflection is selected

Keep `rho` in the existing coordinate units and average the reflected scalar
field on a circle of that radius. The Fourier multiplier remains
`J0(rho*|k|)`, now using the doubled spatial period. The old periodic
gyroaverage must not be reused as the reflected average.

Current axes omit the upper endpoint. A consistent conversion can append its
existing periodic trace without moving the original samples, then form the
doubled reflection. The averaged upper and lower endpoint values can differ:
retain both endpoints or the full doubled block instead of cropping and
silently restoring their old equality. Final representation and public API
will be fixed after the physical choice above.

Record the selected potential extension, GC rotation and display conventions
in new saved data. Missing historical tags continue to mean ordinary
periodicity. Do not resume an old run as a calculation of the new dynamics.
