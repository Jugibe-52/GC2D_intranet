# Eight-method JAX timings

The study compares RK2, ImplicitMidpoint, RK4, GaussLegendre4, ABBA2Midpoint, ABBA2Implicit, BM4Midpoint and BM4Implicit. It uses the original 40 particles introduced in section 4.1, verified as the unchanged subset of the reference campaign, the reflected measured potential at normalized rho=0.015, 500 forcing cycles, 40 fixed steps and ten saved samples per cycle. Each invocation produces 20,000 integration steps and 5,001 saved states per particle. This dedicated runtime study does not alter the 84-particle, 5,000-cycle reference campaign.

The [calculation notebook](../../notebooks/04_04_cout_calcul/calcul.ipynb) executes the eight methods sequentially in one Modal CPU container, with two requested/maximum cores and 4096/8192 MiB requested/maximum memory. JAX uses float64. All implicit methods retain analytic Newton Jacobians, absolute tolerance 1e-12, relative tolerance 1e-11 and at most 40 corrections. The four rounds contain one first invocation and three cached repetitions per method: 32 complete integrations, each of 500 cycles. Method order reverses between successive rounds. CPU sharing outside the container is not controlled, and the processor model may be hidden by Modal.

Timing covers `Execution.run`: preparation, integration, JAX synchronization and construction of host results. It excludes input restoration, container startup, audit checks, file/network transfers and figures. The first invocation includes compilation and other first-use overhead; it is not an isolated measurement of compilation. Persistent JAX caching is disabled, and the in-process integration cache is cleared before the first round. Every first call must register exactly one miss, and every later call exactly one hit. Completed, finite float64 states, aligned outputs, exact step counts and bitwise repeatability are checked outside timing. No inter-particle or inter-method distances, energy observations or optional step histories are computed. These choices follow the [JAX timing guidance](https://docs.jax.dev/en/latest/benchmarking.html).

The [visualization notebook](../../notebooks/04_04_cout_calcul/visualisation.ipynb) verifies saved solution archives and derives medians and minimum–maximum ranges from individual wall-clock times. It exports one French-labelled PDF/PNG figure, `comparacion_tiempos_metodos_fr`, plus JSON provenance into `TFM_redaccion/figuras/`. The lower axis shows absolute first/cached times in seconds; the upper axis shows those same measurements divided by the fastest cached median (RK2). Both axes are logarithmic, with the same fixed denominator for every marker and range endpoint. The thesis presents the combined chart and its analysis without subsections. The three repetitions describe observed variability, not confidence intervals. No accuracy or efficiency-at-matched-error claim follows from this runtime-only experiment.

Both notebooks share `studies/method_timings` and run identifier `jax-20261118-rho0015-c500`. The final data destination is `gc2d_data:gc2d-notebooks-data/studies/method_timings/jax-20261118-rho0015-c500`. The matching directory under `outputs/` is a local recovery cache. Sources, constraints, particle input and reference selection are frozen before submission; the source HDF5 fingerprint is checked. Authenticated execution payloads remain local. A confirmed Modal call receipt prevents duplicate submission after interruption. An unconfirmed receipt requires inspecting Modal before resubmitting. The worker receives no B2 credentials.

Results use the existing solution archive format, including a checksum manifest, and one retained trajectory per method supports completeness checks without distance analysis. All raw scientific and provenance files are uploaded immutably, then downloaded again and verified byte for byte; loaded states and metadata must also agree. Failed publication preserves local data and remains a failure. A new run identifier is required for new measurements. Completed calculations can be recovered and visualized without any numerical execution.

Run the notebooks in order with the project virtual environment. Use reduced cycle counts only in disposable validation copies; the canonical notebooks retain 500 cycles. Study implementation and acceptance are recorded in [the change record](../../specs/changes/2026-11-18-method-timings/results.md).

## Executed study

All 32 integrations completed in Modal on 2026-11-18, call `fc-01M7QB6V4CT4RWB38841X4AY9F`. There were eight first-call compilation cache misses and 24 later hits; every invocation completed 20,000 steps, and repeated states were identical. Python 3.12.10, JAX/jaxlib 0.7.1 and NumPy 1.26.4 were recorded; the CPU model was reported as `unknown`. All 36 raw result/provenance files were downloaded from B2 and verified.

| Method | First invocation (s) | Cached median (s) | Minimum–maximum (s) |
| --- | ---: | ---: | ---: |
| RK2 | 6.037879 | 2.164691 | 2.137034–2.284513 |
| ImplicitMidpoint | 14.050463 | 5.886767 | 5.714194–5.925600 |
| RK4 | 6.790530 | 3.050642 | 2.819637–3.155285 |
| GaussLegendre4 | 16.413041 | 7.656109 | 7.651559–7.755714 |
| ABBA2Midpoint | 7.589155 | 4.342777 | 4.319757–4.632738 |
| ABBA2Implicit | 29.906867 | 24.468711 | 22.446477–24.761951 |
| BM4Midpoint | 22.847080 | 19.328531 | 19.239684–21.751638 |
| BM4Implicit | 101.012471 | 87.321191 | 81.878077–88.005103 |

The fastest cached median is RK2. BM4Implicit takes 40.34 times that median. These figures concern the same container and fixed step, not accuracy at matched computational cost.

The original two figures, their JSON provenance, both executed notebooks and the original visualization source are archived under `figures/v1` in the same study bucket, with nine verified files plus a manifest. Later figure revisions require a new immutable derived prefix.

The combined figure is a local presentation update generated from the same verified measurements; it has not been uploaded to B2. The old relative figure remains as a historical artifact and is no longer exported or included in the thesis.
