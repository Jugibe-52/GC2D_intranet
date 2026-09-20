"""Validate saved/downloaded products without integrating any trajectories."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import study_io
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--staging',action='store_true');args=p.parse_args()
cfg=json.loads((ROOT/'experiment.json').read_text());run='checkpoint_resume_test' if args.smoke else cfg['run_id']
if args.staging:study_io.ROOT=ROOT/'aws/download_staging'
directory,meta,a,positions,initial=study_io.load_calculation(run)
n=len(cfg['particle_ids']);cycles=2 if args.smoke else cfg['cycles'];spc=cfg['steps_per_cycle'];steps=cycles*spc;eq=np.testing.assert_array_equal
for key,value in dict(method=cfg['method'],particle_count=n,cycles=cycles,steps_per_cycle=spc,complete_steps=steps,process_count=16,reference_computed=False).items():assert meta[key]==value,key
assert meta['snapshot_sha256']==study_io.digest(ROOT/'assets/gc2d_snapshot.zip')
manifest=json.loads((ROOT/'aws/input_manifest.json').read_text())
assert meta['checkpoint_driver_sha256']==manifest['source_sha256']['checkpoint_calculation.py']
original=json.loads((ROOT/'assets/original_particles.json').read_text())['particles'];ids=np.asarray(cfg['particle_ids'])
eq(a['particle_ids'],ids)
eq(a['initial_positions'],[[original[i-1]['x'],original[i-1]['y']] for i in ids])
eq(a['colors_rgba'],[original[i-1]['rgba'] for i in ids]);eq(a['colors_hex'],[original[i-1]['color'] for i in ids])
for key,value in a.items():
 if value.dtype.kind in 'if':assert np.isfinite(value).all(),key
assert a['states'].shape==(2*n,steps+1) and a['cycle_positions'].shape==(cycles+1,n,2)
eq(a['times'],np.arange(steps+1)*(1/spc));eq(a['cycle_times'],np.arange(cycles+1))
eq(a['cycle_positions'][:,:,0],a['states'][:n,::spc].T);eq(a['cycle_positions'][:,:,1],a['states'][n:,::spc].T)
assert len(positions)==n*cycles and len(initial)==n
eq(positions['particle'],np.tile(ids,cycles));eq(positions['cycle'],np.repeat(np.arange(1,cycles+1),n));eq(positions['color'],np.tile(a['colors_hex'],cycles))
eq(positions[['x_unwrapped','y_unwrapped']],a['cycle_positions'][1:].reshape(-1,2))
for w,worker in enumerate(meta['workers']):
 assert worker['particle_ids']==ids[w::16].tolist()
 assert worker['new_steps']+worker['resumed_steps']==steps
if cfg['method']=='BM4Implicit':
 for key in ('nonlinear_iterations','nonlinear_residuals','nonlinear_tolerances','projection_multiplier_norms'):assert a[key].shape==(16,steps)
 offsets=a['newton_history_offsets'];eq(np.diff(offsets,axis=1),a['nonlinear_iterations']+1)
 eq(offsets[1:,0],offsets[:-1,-1]);assert offsets[0,0]==0
 residual=a['newton_history_residuals'];mu=a['newton_history_mu_norms'];assert len(residual)==len(mu)==offsets[-1,-1]
 ends=offsets[:,1:].ravel();eq(residual[ends-1],a['nonlinear_residuals'].ravel());eq(mu[ends-1],a['projection_multiplier_norms'].ravel());assert (a['nonlinear_residuals']<=a['nonlinear_tolerances']).all()
 expected=[('newton_steps.csv.gz',16*steps),('newton_iterations.csv.gz',len(residual))]
else:
 assert a['copy_separation_norms'].shape==(16,steps) and not meta['newton_history_recorded']
 assert not any('newton' in key or 'multiplier' in key for key in a)
 expected=[('midpoint_steps.csv.gz',16*steps)]
for filename,count in expected:assert sum(len(frame) for frame in pd.read_csv(directory/filename,chunksize=100000))==count
for category,notebook in [('resultados','calculo_executed.ipynb'),('figuras','visualizacion_executed.ipynb')]:
 folder=study_io.ROOT/category/run;status=json.loads((folder/'execution_status.json').read_text());assert status['status']=='success'
 for cell in json.loads((folder/notebook).read_text())['cells']:
  if cell['cell_type']=='code' and ''.join(cell['source']).strip():assert cell['execution_count'] is not None
  assert not any(o.get('output_type')=='error' for o in cell.get('outputs',[]))
report=dict(verified_utc=study_io.utc_now(),run_id=run,method=cfg['method'],all_numeric_manifest_hashes_verified=True,original_ids_positions_colors_preserved=True,return_position_rows=len(positions),workers=16,complete_steps=steps,executed_notebooks_successful=2,external_archive_checksum_verified=False)
study_io.atomic_json(ROOT/'aws'/('validation_smoke_products.json' if args.smoke else 'validation_download_internal.json'),report)
print(json.dumps(report,indent=2))
