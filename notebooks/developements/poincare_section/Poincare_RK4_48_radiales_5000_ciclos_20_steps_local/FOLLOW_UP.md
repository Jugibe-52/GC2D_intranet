# Completed and verified

Local RK4 finished successfully on 2026-09-20 at 17:16:25 UTC. Products and executed notebooks were validated, with 240,000 return positions. Completion is being notified. Do not restart or re-notify this study. Keep the shared monitor active for BM4 implicit.

# Monitor the authorized local RK4 run

Read local_status.json, launch_record.json and resultados/local_rk4_48p_5000c_20s_20260920/checkpoints/worker_*/progress.json. This study runs locally with four processes, 48 original particles, 5,000 cycles and 20 steps/cycle. Do not create or modify AWS resources. The heartbeat also follows the independent BM4 implicit AWS study. Preserve that monitoring when this local study completes.

The background local_job.py calculates, visualizes and validates automatically. local_status.json must say success, stage verified, before reporting completion. Check validation/local_rk4_48p_5000c_20s_20260920_products.json for 240,000 return-position rows, preserved IDs and colors and four workers. Verify the static Poincare plot and that the interactive selector exists in figuras/<run-id>/poincare_selector.html. All outputs remain in this folder. No Newton, multiplier or reference data apply.

Stay quiet while ordinary progress continues. On success, provide the result folder and interactive selector links and mark this local study complete. The shared heartbeat must remain active while the independent BM4 implicit study is pending. On failure or missing process with stale status, inspect local_job.log and the last execution_status.json, preserve checkpoints and report the actual problem. Do not duplicate a running job. Safe resumption uses the same run ID via local_job.py after checking the existing process and lock. A sleep inhibitor is active only for the lifetime of the job; do not shut down the user's computer.
