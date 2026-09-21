# BM4Implicit Poincare experiment

Independent reproducible study with 24 original particles, 5,000 cycles, 40 complete steps per cycle and 16 processes. Particle IDs and colors are inherited from `assets/original_particles.json`; IDs are never renumbered. No reference is computed.

Run `python run.py calculate --run-id aws_bm4implicit_24odd_5000c_40s_20260920 --resume --timeout 28800`, then `python run.py visualize --run-id aws_bm4implicit_24odd_5000c_40s_20260920`. Use `--smoke` with a new run ID for two-cycle validation. Numerical outputs are in `resultados/<run-id>` and figures and the offline particle selector in `figuras/<run-id>`.

The checked-in reusable runtime source is under `src/studies/poincare_batch_runtime` and plotting source under `src/visualization`. The local files here are deployment copies frozen and checksummed with the input package. The original scientific source snapshot is unmodified. Checkpoints commit every 1000 steps; restart uses original global times. For implicit runs, all Newton residuals and multiplier infinity norms per group are saved. Midpoint runs record copy separation and have no Newton or implicit multiplier records.

AWS uses one c8a.4xlarge Spot instance, a $0.50/hour maximum price, encrypted storage and a 8-hour global deadline including interruptions. Results and checkpoints are uploaded to private S3; the instance stops after successful export or failure.
