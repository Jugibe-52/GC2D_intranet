# Local classical RK4 study

48 original particles, 5,000 cycles, 20 steps/cycle, four processes. Parameters are explicit in calculo.ipynb. Run the calculation notebook, then visualizacion.ipynb to load saved data without reintegrating. Original IDs, positions and colors are retained. No reference is requested and no Newton/mu diagnostics apply to RK4.

Use `python local_job.py` to calculate, validate and visualize. Restart uses the same run ID and verified immutable checkpoints every 25 cycles. Calculation data and executed notebooks are in resultados/<run-id>; plots and the interactive particle selector are in figuras/<run-id>. The trajectory archive is named rk4_trajectory.npz; metadata explicitly identifies RK4. Checkpoint arithmetic follows fixed public-RK4 chunks and is identical across restarts.

Runtime sources live in src/studies/poincare_batch_runtime and plotting sources in src/visualization. Local copies are frozen for this run. The shared original scientific snapshot is unchanged. All scientific data and output paths are relative to this study folder.
