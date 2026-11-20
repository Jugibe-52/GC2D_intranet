# Accepted behavior

Keep the existing 84 physical initial positions, eight methods, eleven radii from 0 to 0.50, 5000 cycles, 40 fixed steps per cycle and ten aligned saved samples per cycle. The requested rho is in base-cell widths and reaches the gyroaverage unchanged. Keep the reflected field, B=1.5 and existing Newton controls.

Record physical_hamiltonian and its analytic partial time derivative directly; energy_drift is H-H0, not a conservation error. Remove energy_integral, energy_balance and energy_quadrature_difference from new results. Retain signed projection diagnostics. Historical archives remain readable and unchanged.

Use a distinct main-20261116-direct-rho campaign directory and matching B2 prefix. Check inputs and short representative runs before submitting the full campaign. Update production at the existing Cloudflare Pages project from verified new results. Update chapter 4 notebooks and generated figures using the corrected convention, preserving particle identities and historical provenance; reassess classifications and zooms rather than assuming they survive the radius correction.
