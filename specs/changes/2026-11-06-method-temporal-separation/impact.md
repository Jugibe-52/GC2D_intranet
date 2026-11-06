# Affected parts

Change the numerical preparation protocol, methods/base, classical and extended initialization, derivative selection, host execution, SciPy control and JAX preparation. Migrate direct preparation calls and tests; simulate(problem, method) remains the public complete-run entry point. Existing configuration options and result formats remain available. Recreate previously serialized prepared runs; they contain obsolete execution state.

Update current simulation contracts, relevant architecture and migration documentation, repository memory and chapter 3 of the sibling TFM_redaccion repository. Keep Spanish reading copies. Search supported notebooks for direct preparation calls without running long studies. Preserve pre-existing edits in both repositories, including the Hamiltonian changes.

Check differentiation errors, deferred evaluations, independent runs, nonzero starting times, endpoint clipping, shadow samples, adaptive interpolation statistics, passive energy and host/JAX agreement. Compare small trajectories and work counts with snapshots captured before implementation.
