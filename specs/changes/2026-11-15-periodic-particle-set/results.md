# Execution results

The user accepted this search and verification scope on 2026-11-15 with “& empieza”. Eight distinct, nonstationary, zero-winding periodic initial conditions are now saved as IDs 85–92 in `data/particles/periodic_8.json`, at rho = 0.30. Their numerically resolved minimum periods are 5 through 12 forcing cycles. The simple notebook loads all four populated datasets and has been executed successfully.

## Measured checks

All reference files passed their SHA-256 and byte-count checks. Candidate extraction from 5001 integer-cycle samples of 84 unwrapped trajectories reproduced byte-for-byte: 198 seeds were prepared, and 12 RK4 period searches found 25 candidate orbits. Final cross-orbit checks reject cyclic phase duplicates and period-two translations.

Errors below are Euclidean distances in units of the base cell width. Every earlier integer-cycle return remains more than 1e-4 from its initial position. Final Gauss grids follow at least 80, 160 and 320 steps per cycle, extended as needed; each successive final seed shift is below 1e-7.

| ID | Period (cycles) | Final Gauss steps/cycle | Check steps/cycle | Fine Gauss closure | Independent RK4 closure | Maximum over 5 repeated periods |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 85 | 5 | 320 | 640 | 3.156e-10 | 3.527e-10 | 4.727e-08 |
| 86 | 6 | 640 | 1280 | 7.121e-10 | 4.916e-10 | 9.285e-08 |
| 87 | 7 | 320 | 640 | 1.381e-09 | 7.328e-10 | 1.486e-07 |
| 88 | 8 | 640 | 1280 | 1.466e-10 | 1.543e-10 | 2.167e-07 |
| 89 | 9 | 640 | 1280 | 1.438e-10 | 3.514e-11 | 1.535e-07 |
| 90 | 10 | 1280 | 2560 | 2.744e-10 | 2.688e-09 | 3.621e-08 |
| 91 | 11 | 320 | 640 | 7.550e-10 | 6.359e-10 | 6.502e-07 |
| 92 | 12 | 320 | 640 | 1.873e-09 | 1.821e-09 | 3.498e-07 |

The first period-11 candidate failed the five-period threshold even after refinement and was excluded. Its second candidate passed without relaxing the tolerances. Initial period-6, 8 and 9 checks also failed at the coarser grid and passed after continued refinement. Failed attempts remain recorded. One infrastructure preemption occurred; lost work is not included in the measured completed-worker estimate.

## Resources and scope

Modal/JAX float64 executed 25 recorded search or verification tasks, each requesting 2 physical CPU cores and 4 GiB RAM with an 8 GiB limit. Summed completed-worker wall time is 1747.110 seconds; this is not elapsed wall time because tasks overlap and stages are separated. Estimated compute is USD 0.061289 under requested resources, or USD 0.076803 under resource limits. These estimates exclude startup, idle time, transfer, preempted work and invoice adjustments. Per-task times, resource details and call IDs are retained in the JSON.

Five focused tests passed, including nonfinite-return rejection. Mypy passed for all 324 source files. The complete lightweight three-code-cell notebook executed successfully, with four populated panels. The dataset finalizer checked all eight records and their pairwise distinctness before writing.

The three earlier particle datasets, the main campaign and the thesis are unchanged. Nothing from this study was uploaded to B2. Numerical verification is limited to the stated potential, rho, phase and finite resolutions; it is not a rigorous existence proof or a claim of long-time stability. SDD structure validation passed: 209 records and 22 requirements.
