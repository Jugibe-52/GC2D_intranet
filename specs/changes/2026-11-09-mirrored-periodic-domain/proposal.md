# Selected construction

Add spatial_extension to Potential and its factories, inherited by JaxPotential.
Its default is "reflected"; "periodic" preserves current interpolation and
averaging. Keep Grid's half-open samples and base width L. Expose spatial_period
as 2*L or L, respectively.

For reflected fields fit cubic tensor splines with zero first derivative at both
ends of each axis, independently for mean and complex harmonic components.
This gives a C2 reflected scalar and C1 canonical field. Reflected mode requires
interpolation_order=3; periodic mode retains degrees 2 through 5. Other reflected
degrees need a separately specified boundary construction.

Keep read-only reflected_samples of shape (1+mode_count,nx+1,ny+1), also accepted
as an optional constructor argument. Its first field is the real mean; its
interior must match mean/modes. When absent, complete missing upper endpoints
using the existing periodic trace without moving source samples or changing L.
This is an input completion convention, not a Neumann condition or a requirement
that opposite edge values always agree.

Fold each coordinate with u=(x-x0) % (2*L) and
x_m=x0+where(u<=L,u,2*L-u). Derivatives (a,b) receive sx**a*sy**b.
Use consistent CPU/JAX branch signs at seams. Obtain derivatives from constrained
splines without masks that destroy automatic derivatives. GC keeps
f=(-H_y,H_x), H=Phi_rho/B and Hessian-based Jacobians. Normal velocity may cross
the base edges; tangential velocity vanishes.

For rho>0, reflect closed samples into a doubled half-open FFT block, multiply
by J0(rho*|k|) using that block's angular wave numbers, retain the entire closed
base block after inverse FFT, and refit constrained splines. This is sampled
spectral averaging followed by spline reconstruction, not exact circle
integration of the cubic spline. Keep radius units and temporal frequencies.
Scaling, averaging and persistence preserve independent upper edges.

Provide display_coordinates and display_vectors helpers. Fold display copies
into the base square and transform actual trajectory velocities componentwise.
Label canonical base-tile backgrounds: opposite folded branch orientations can
have opposite arrow directions. Distances use spatial_period; base placement
and axis normalization keep L. Saved states stay unwrapped.

Persist extension and closed samples. Missing historical tags mean periodic;
retain historical fingerprints and archive keys. New study configurations record
their choice and reject incompatible resumes. Do not edit TFM_redaccion, old
result files, or numerical methods; no commits, pushes or publication.
