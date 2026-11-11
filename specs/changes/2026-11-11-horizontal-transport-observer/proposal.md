# Accepted calculation

Compute Q_x = integral_0^N |dx/dt| dt / (N W_x), with N=5000 unit-duration cycles and W_x=max(x)-min(x). Use physical unwrapped x. Keep the historical gyro-radius conversion and radial Hamiltonian unchanged.

A passive observer applies composite trapezoidal quadrature to actual |vx| at every accepted full-step endpoint and tracks extrema including the initial point. Poincare returns and signed splitting substeps are not quadrature nodes. Zero width yields an undefined normalized value. Keep bounded per-particle JAX accumulators and check quadrature refinement.

Repeat the original star and probe groups separately in Modal, retain cycle histories and observer summaries, update the notebook and show four values beside each particle at the selected rho. Publish the existing Cloudflare site using the new trajectories and summaries together; preserve the old publication locally.
