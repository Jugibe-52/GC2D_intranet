# Agreed experiment

Use PHI_2.h5, constant field 0 and temporal mode 15, B=1.5, no smoothing, cubic reflected spatial extension, and the existing physical-cycle normalization. Keep rho=0,0.05,...,0.50 with physical radius rho*0.06/(2*pi). Copy the exact 84 initial positions, IDs and colors from the existing particle record, retaining its fingerprint.

Compare ABBA2Midpoint, ABBA2Implicit, BM4Midpoint, BM4Implicit, ImplicitMidpoint, RK2 (explicit midpoint), RK4 and GaussLegendre4. Use 5000 cycles, 40 fixed steps/cycle and ten aligned saved samples/cycle. This is 88 cases and 50001 saved states per particle, including the initial state. Use JAX float64 on Modal CPU with identical analytic Newton controls (absolute 1e-12, relative 1e-11, at most 40 corrections).

Save unwrapped states, H, analytic partial_t H, the accumulated endpoint quadrature and its energy balance. For the two implicit projection methods, save signed per-particle multipliers, endpoint infinity norms and maxima over each four-step recording block. Retain nonlinear-work diagnostics and provenance. Quadrature error is separate from integration error and will be checked on reduced runs.

Store versioned, verified archives in Backblaze B2 with local recovery and Modal receipts. Create a separate Cloudflare Pages viewer with synchronized rho/particle/cycle controls, showing only one Poincare return per cycle. Exclude intermediate trajectory samples, energy/projection plots and horizontal transport Qx. Respect the reflected field's spatial period two; a period-one wrap is invalid.

The user explicitly requested preparation and execution of this previously agreed scope on 2026-11-15, including Modal, B2 and Cloudflare. These requested remote operations are part of the experiment; Git commits and pushes are not. Pilot checks precede production.

Each rho/method case also records measured worker wall/CPU time, allocated
physical cores and RAM, visible CPU model/count, software versions and peak RSS.
A versioned public tariff gives requested/maximum-memory compute estimates in
USD; actual billing stays explicitly unknown. These observations do not modify
the trajectories. Cases run independently with up to sixteen Modal workers.

The interface refinement accepted on 2026-11-15 uses a compact white layout with all eight methods permanently visible, without a method selector. Desktop order is RK2, ImplicitMidpoint, RK4, GaussLegendre4 above ABBA2Midpoint, ABBA2Implicit, BM4Midpoint, BM4Implicit. Two full-width sliders select the start/current cycles with one-cycle minus/plus buttons and a shared interval indicator. Moving an endpoint past the other moves both to that cycle. Reset restores the full interval; Full view restores the shared `[0,1] × [0,1]` viewport, also used on initial load. The accepted unit-square display clips outside returns without changing the period-two coordinates. Narrow screens use two or one columns.

## Accepted partial publication

The user accepted partial publication on 2026-11-15. Add an explicit allow-partial publication option that admits only B2-verified cases and recorded failures without a saved archive. Keep the default complete-campaign requirement. Export 5001 whole-cycle returns per particle for each available case, preserve eight panels, and label the two missing panels at rho=0 with their failure status. Show campaign availability and continue loading other panels when one is absent or fails to load. Deploy to the existing Cloudflare Pages production branch at https://gc2d-main-reflected.pages.dev/. Preserve the pilot and the thesis text.
