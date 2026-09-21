# Completed on 2026-09-20T14:19:22.829543+00:00

All finalization steps are complete. Do not restart the instance or calculation. See `validation_download_final.json`, `cloud_status_final.json`, `spot_cancelled_dom.txt` and `instance_stopped_final_dom.txt`. The heartbeat is paused.

# Finish the authorized 48-particle Spot study

The user authorized implementation, tests, AWS launch and downloading positions, multiplier norm evolution and Newton convergence. Do not start a separate study. Preserve 48 fresh radial particles, 5,000 cycles, 20 steps/cycle, 16 workers and no reference.

Local folder: `/home/juan/Proyectos/GC2D_intranet/notebooks/developements/poincare_section/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot`.
Read `aws/launch_record.json` and `aws/input_manifest.json` for verified deployment data.

- Instance `i-0180fedd146c314c1`, c8a.4xlarge Spot, Frankfurt.
- Service `poincare-48p-5000cycles.service`.
- Persistent Spot request `sir-h4ezpjfg` (only this one may be cancelled).
- Global VM shutdown deadline: 2026-09-20 16:00:31 UTC.
- Follow-up automation ID: `descargar-estudio-bm4-de-48-part-culas`.
- Run `aws_48p_5000c_20s_16proc_spot_20260920`.
- Bucket `gc2d-frankfurt-20260904-j7m3q9`, prefix `runs/aws_48p_5000c_20s_16proc_spot_20260920/`.
- EC2 detail URL: https://eu-central-1.console.aws.amazon.com/ec2/home?region=eu-central-1#InstanceDetails:instanceId=i-0180fedd146c314c1
- Private status: `cloud_status.json`, `cloud_progress.json`, `calculation.log`, `checkpoints/worker_XX/chunk_XXXXXXXX.{json,npz}`.
- Remote study folder is `/home/ubuntu/poincare/` followed by the same local folder basename.

Use the Chrome skill and the user's AWS console session. Do not extract credentials or use local AWS CLI access as a browser workaround. SSM console commands are allowed for reading this study's service/logs and managing its already-authorized execution. Mark the useful AWS tabs for handoff. Browser actions may open new tabs: inspect the owned-tab list and use the actual new tab.

While the calculation advances normally, remain quiet. Check for failed/stale progress or Spot interruption. The persistent Spot request uses stop on interruption, and the enabled system service resumes the same verified checkpoints on reboot. A six-hour global timer survives boots. Do not extend the limit or create additional paid instances automatically. The request expires at 2026-09-20 18:00 UTC; resolve any stopped incomplete run by preserving checkpoints and informing the user.

On completion, the cloud job uploads an archive and SHA256 sidecar, then shuts down the VM. Download from the private S3 console through the supported download event workflow. Register the download listener before the Download click and attach a rejection handler immediately. Chrome saves files in `/home/juan/Descargas`; inspect the actual downloaded filenames. Download cloud status too.

Verify the archive checksum against its sidecar and cloud status. Reject absolute paths, traversal, links and unexpected archive members. Extract into a temporary staging folder, then place `resultados/<run-id>/`, `figuras/<run-id>/` and `resultados/latest_success.json` in this study folder. Retain existing smoke results. Load with `study_io.load_calculation(run_id)` to verify all seven numeric file hashes.

Validate metadata 48 particles, 5,000 cycles, 20 steps/cycle, 16 workers, no reference, full Newton history, 100,000 complete steps, original source hashes. Check all states finite, shape (96,100001), cycle positions (5001,48,2), 240,000 return CSV rows, 48 initial positions and unique fixed colours. Verify diagnostic shapes (16,100000), offsets (16,100001), each history count equals corrections+1, and final residuals meet their own tolerances. Check step CSV has 1,600,000 rows and all iteration rows agree with the saved arrays. Verify successful executed notebooks and inspect plots. Saved histories use infinity norm of the reduced multiplier per three-particle group, not a separately computed norm per particle.

Record verified download and final instance state. Cancel ONLY this study's persistent Spot request once finished; if still running, cancel the request without terminating the instance, then stop it and verify Stopped. Never terminate before cancelling a persistent request. Preserve EBS/S3 results. Do not affect older stopped instances or other work. Notify the user with concise Spanish links to both notebooks, position CSV, Newton step/history data and figures. Pause this study's heartbeat after successful download and cleanup. If blocked by lost Chrome authorization, preserve progress, explain the actual blocker and do not claim a completed download.
