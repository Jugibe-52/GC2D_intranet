# Execution record

Accepted and authorized on 2026-11-11: repeat the complete integration with an observer, all original cases, update the notebook and website, maximum Modal expense EUR 15.

## Measurement and provenance

The observer accumulates trapezoidal quadrature of absolute physical x velocity at every accepted complete step. It includes the initial state and all accepted endpoints in the x extrema. The reported quantity is Qx = integral(abs(vx)) / (5000 * (xmax - xmin)), without angular distances or an extra 2pi factor. This is a finite-resolution numerical integral. Zero width is represented as undefined, never silently replaced with zero.

All 88 archived metadata records passed fingerprint, geometry and initial-state checks. The original 40-star and 8-probe integration groups, radial Hamiltonian, source field, historical clock, 5000 cycles, 50 steps per cycle and Newton controls are preserved. The original archives remain intact. New metrics accompany the trajectories from the same repeated integration.

An independent comparison at rho=0.3 verified original and new archive hashes for all four methods. Maximum differences in the first cycle range from 5.55e-17 to 7.77e-16 m; through cycle 10 they remain below 1.31e-14 m. Material differences appear by cycle 100, consistent with chaotic amplification of rounding differences. Bitwise reproduction of long trajectories is not claimed.

Sixteen Modal benchmark/refinement jobs completed. At rho=0.3 over one cycle, maximum integral differences between 50 and 200 steps per cycle are below 0.187%. This is a short validation, not a global error bound for the 5000-cycle result.

## Validation

The complete regression suite ran 765 tests in 1093.266 seconds: 762 passed and 3 were skipped. Type checking passed for 316 source files. Both source and wheel packages built successfully. Focused checks cover analytic motion, quadrature refinement, observer passivity, CPU/JAX agreement for all four methods, Modal serialization, reproduction provenance and viewer validation. Specification validation passed with 199 records and 22 requirements; the Git whitespace check is clean.

## Completion and publication status

All 88 jobs completed without failure. The independent archive audit verified all 2112 finite values, 250000 accepted steps and 5000 cycles per case, the saved file hashes, particle IDs and exact normalization identity; there are no zero-width cases. The main batch elapsed time was 31.65 minutes, with a measured peak of 44 concurrent workers. Combined main/benchmark worker time was 10.1131 hours. Multiplying this by the configured upper CPU/memory resource rate gives a compute estimate of USD 1.28032, excluding startup, idle time, image build and egress. The final API snapshot records USD 1.17499503 before credits and USD 0 billed; reporting may still update. These are distinct provisional measurements, not a final invoice. Resource/time ceilings and the reserved overhead kept this run within the accepted EUR 15 budget.

The notebook's setup and inspection cells executed successfully and their actual outputs are saved. The viewer contains new Poincare trajectories paired with all 2112 full-integration metrics. All eleven rho selections, 48 particle cards and 192 method values per selection passed real-data browser inspection. Changing the cycle interval leaves metrics fixed; hide/show selection, tooltips and field phase controls work. No browser console errors were observed. Packaging round-trips the complete payload exactly and every asset is below the Pages size limit.

The notebook directory contains horizontal_transport_results.json, the updated visualisation.ipynb and four_methods_poincare.html. Durable receipts, saved solutions, source fingerprints, scientific checks and billing snapshots are retained in horizontal_transport/. Full-integration metrics can exceed one; they are neither probabilities nor percentages.

After user-completed OAuth authorization, the update was published to the existing Cloudflare Pages project on 2026-11-11. The production URL is https://gc2d-poincare-four-methods.pages.dev/ and the deployment URL is https://92f49b9a.gc2d-poincare-four-methods.pages.dev. Cloudflare confirmed all 18 uploaded content files plus the headers configuration. The production browser rendered 48 particle cards and 192 values for each of all eleven rho selections; text fingerprints and lengths match all 2112 local formatted values. No browser console errors were observed. Direct Python HTTP requests received Cloudflare 1010/403, so verification uses the successful live browser inspection rather than claiming public asset-hash checks.

cloudflare_pages/deployment.json now records deployment 92f49b9a and the source/result hashes. cloudflare_pages/pending_deployment.json retains the asset manifest with status published_verified; the previous receipt is preserved in horizontal_transport/original_deployment.json. The notebook and documentation now report publication complete. No simulation was repeated for publication.
