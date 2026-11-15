"""Poincare-only static export with lazy, lossless binary assets per case."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import numpy as np

from diagnostics.io.persistence import load_solution
from studies.main_campaign import validate_case, CampaignConfig


def poincare_returns(solution: Any, metadata: dict[str, Any]) -> np.ndarray:
	"""Select the saved whole-cycle nodes; map into the true period-two cell."""
	validate_case(solution, metadata)
	config: CampaignConfig = CampaignConfig(**metadata['config'])
	indices: np.ndarray = np.arange(config.cycles + 1) * config.samples_per_cycle
	if not np.allclose(solution.t[indices], np.arange(config.cycles+1)*metadata['cycle_duration'], rtol=0., atol=1e-10):
		raise ValueError('Saved Poincare samples do not have one fixed forcing phase.')
	states: np.ndarray = solution.states[indices]
	# Shape (cycle, particle, xy). No interpolation, trajectory replay or downsampling.
	xy: np.ndarray = np.stack((states[:, :84], states[:, 84:]), axis=-1)
	return np.ascontiguousarray(np.mod(xy, metadata['spatial_period']), dtype='<f8')


def export_campaign(directory: Path, *, allow_partial: bool = False) -> Path:
	"""Publishable files contain only whole-cycle returns, never ten-sample histories."""
	record: dict[str, Any] = json.loads((directory / 'campaign.json').read_text())
	missing: dict[str, Any] = {}
	if allow_partial:
		from studies.main_campaign_io import publication_cases
		_, missing = publication_cases(directory, record, allow_partial=True)
	output: Path = directory / 'site'
	assets: Path = output / 'assets'
	assets.mkdir(parents=True, exist_ok=True)
	cases: dict[str, Any] = {}
	for rho in record['rhos']:
		for method in record['methods']:
			key: str = f'rho-{rho:.2f}-{method}'
			if key in missing:
				cases[key] = missing[key]
				continue
			path: Path = directory / 'runs' / key
			if not (path / 'manifest.json').exists():
				raise ValueError(f'Missing case: {key}; production export requires a complete campaign.')
			stored: Any = load_solution(path)
			if (stored.metadata['config'] != record['config'] or stored.metadata['method'] != method
					or stored.metadata['rho'] != rho or stored.metadata['code_sha256'] != record['code_sha256']
					or stored.metadata['particle_record'] != record['particles']):
				raise ValueError(f'Campaign provenance mismatch: {key}.')
			xy: np.ndarray = poincare_returns(stored.solution, stored.metadata)
			payload: bytes = xy.tobytes()
			digest: str = sha256(payload).hexdigest()
			name: str = f'{key}-{digest[:12]}.bin'
			if len(payload) > 25*1024*1024:
				raise ValueError('A case exceeds the Cloudflare Pages per-file limit.')
			(assets / name).write_bytes(payload)
			cases[key] = dict(status='available', asset=f'assets/{name}', sha256=digest, bytes=len(payload),
				shape=list(xy.shape), rho=rho, method=method)
	# Compare the four Runge--Kutta recipes above the four palindromic recipes.
	display_order: tuple[str, ...] = ('RK2', 'ImplicitMidpoint', 'RK4', 'GaussLegendre4',
		'ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit')
	config: dict[str, Any] = dict(cycles=record['config']['cycles'], rhos=record['rhos'],
		methods=[name for name in display_order if name in record['methods']], particles=record['particles']['particles'],
		families=record['particles']['families'], cases=cases, spatialPeriod=2,
		samplesPerCycle=1, source_sha256=record['source_sha256'], code_sha256=record['code_sha256'],
		publicationStatus='partial' if missing else 'complete', availableCases=len(cases)-len(missing),
		expectedCases=len(cases), missingCases=list(missing))
	(output / 'data.json').write_text(json.dumps(config, separators=(',', ':'))+'\n')
	(output / 'index.html').write_text(_HTML, encoding='utf-8')
	(output / '_headers').write_text('/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n/data.json\n  Cache-Control: no-cache\n')
	return output


_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GC2D · Poincaré sections</title>
<style>
:root{font:14px Arial,sans-serif;color:#263744;background:white;--accent:#087bff}
*{box-sizing:border-box}body{margin:16px}header,main{max-width:1800px;margin:auto}
h1{font-size:20px;margin:0 0 14px}header p{line-height:1.4;margin:0 0 14px}
.bar,.playback,.actions,.families{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
label{display:inline-flex;gap:6px;align-items:center}input,select,button{font:inherit}
button,select{border:1px solid #999;border-radius:2px;padding:6px 11px;background:#f3f3f3;color:#263744}
select{background:white}button{cursor:pointer}button:hover:not(:disabled){background:#e7eff8}button:disabled{opacity:.4;cursor:default}
input[type=range],input[type=checkbox]{accent-color:var(--accent)}
.cycle-controls{display:grid;grid-template-columns:max-content 34px minmax(80px,1fr) 34px;gap:5px 8px;align-items:center;flex:1;min-width:260px}
.cycle-controls input{width:100%;min-width:0;margin:0}.cycle-controls button{padding:4px;font-size:18px;height:29px}
#interval{font-weight:bold;white-space:nowrap;font-variant-numeric:tabular-nums}
#plots{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px 14px}
.panel{min-width:0}.panel h2{font-size:16px;text-align:center;margin:7px 0 5px}
.panel-note{margin:0 10px 6px;text-align:center;color:#864a12;font-size:13px;line-height:1.4}
#availability{margin:0 0 12px;padding:9px 12px;background:#fff4de;border-left:3px solid #bd7a21;line-height:1.4}
canvas{display:block;width:100%;aspect-ratio:1;touch-action:none;cursor:crosshair}
details{margin:16px 0;padding-top:12px;border-top:1px solid #e2e9ed}summary{cursor:pointer;font-weight:bold}
#particles{display:grid;grid-template-columns:repeat(auto-fill,minmax(85px,1fr));gap:6px;margin-top:12px}
#particles label{border:1px solid #dce4e8;border-left:5px solid var(--particle);padding:6px}
.families{margin-top:12px}#status{font-size:13px;color:#526978}.legend,footer{font-size:13px;color:#526978;line-height:1.4}footer{padding:16px 0}
@media(max-width:900px){#plots{grid-template-columns:repeat(2,minmax(0,1fr))}.playback{align-items:center}#interval{width:100%;text-align:right}}
@media(max-width:520px){body{margin:10px}#plots{grid-template-columns:1fr}.cycle-controls{order:2;flex-basis:100%}#interval{order:3}}
</style>
<header><h1>Poincaré sections</h1><p class="legend">Hollow circles: positions at the start cycle. Small dots: accumulated returns. Large dots: current return. Drag to zoom all panels; Shift-drag to pan. Initial view: [0,1] × [0,1]. Coordinates use the true spatial period 2.</p></header>
<main>
<div class="bar"><label>Gyro-radius ρ <select id="rho" aria-label="Gyro-radius"></select></label><span id="subtitle">Loading campaign…</span></div>
<p id="availability" hidden></p>
<div class="playback"><button id="play">Play</button><button id="reset-cycles">Reset</button>
<div class="cycle-controls">
<label for="start">Start cycle</label><button id="start-previous" aria-label="Previous start cycle">−</button><input id="start" type="range" min="0" step="1" value="0"><button id="start-next" aria-label="Next start cycle">+</button>
<label for="cycle">Current cycle</label><button id="cycle-previous" aria-label="Previous current cycle">−</button><input id="cycle" type="range" min="0" step="1" value="0"><button id="cycle-next" aria-label="Next current cycle">+</button>
</div><output id="interval" for="start cycle"></output></div>
<div class="actions"><button id="reset">Full view</button><span id="status" role="status">Loading Poincaré returns…</span></div>
<div id="plots"></div>
<details><summary id="particleSummary">Particles · 84 selected</summary><div class="families"><button id="all">Select all</button><button id="none">Clear</button><span id="families"></span></div><div id="particles"></div></details>
<footer>Source: PHI_2.h5 · reflected cubic spline · B = 1.5 · ρ<sub>m</sub> = ρ × 0.06 / (2π). <a href="data.json">Viewer metadata</a></footer>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
let cfg,selected=new Set(),panels=[],cache=new Map(),view={x:0,y:0,s:1},playing=false,frame=0,loading=0,paint=0;
function setStatus(s){$('status').textContent=s}
function selectedRho(){return Number($('rho').value).toFixed(2)}
function stop(){playing=false;$('play').textContent='Play';cancelAnimationFrame(frame)}
function bounds(){const end=Math.max(0,Math.min(cfg.cycles,Math.trunc(Number($('cycle').value)||0)));const start=Math.max(0,Math.min(end,Math.trunc(Number($('start').value)||0)));$('start').value=start;$('cycle').value=end;$('interval').textContent=`Cycles ${start}–${end} / ${cfg.cycles}`;for(const [id,value] of [['start',start],['cycle',end]]){$(`${id}-previous`).disabled=value===0;$(`${id}-next`).disabled=value===cfg.cycles}return {start,end}}
// Moving either endpoint keeps a valid shared interval, including a single cycle.
function setCycle(id,value){stop();$(id).value=Math.max(0,Math.min(cfg.cycles,value));if(Number($('start').value)>Number($('cycle').value))$(id==='start'?'cycle':'start').value=$(id).value;bounds();cancelAnimationFrame(paint);paint=requestAnimationFrame(render)}
function render(){const {start,end}=bounds();$('particleSummary').textContent=`Particles · ${selected.size} selected`;for(const panel of panels){panel.canvas.setAttribute('aria-label',`${panel.method} Poincaré section at rho ${selectedRho()}, cycles ${start}–${end}${panel.message?': '+panel.message:''}`);draw(panel,start,end)}}
function draw(panel,start,end){const c=panel.canvas,ctx=c.getContext('2d'),size=600,left=53,top=18,w=525;ctx.clearRect(0,0,size,size);ctx.fillStyle='white';ctx.fillRect(0,0,size,size);ctx.font='12px system-ui';ctx.strokeStyle='#e2e9ed';ctx.lineWidth=1;
for(let i=0;i<=4;i++){let x=left+w*i/4,y=top+w*i/4;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+w);ctx.moveTo(left,y);ctx.lineTo(left+w,y);ctx.stroke();ctx.fillStyle='#526978';ctx.textAlign='center';ctx.fillText((view.x+view.s*i/4).toFixed(3),x,top+w+20);ctx.textAlign='right';ctx.fillText((view.y+view.s*(1-i/4)).toFixed(3),left-7,y+4)}ctx.fillStyle='#182b3a';ctx.textAlign='center';ctx.fillText('(R − R₀) / L',left+w/2,591);ctx.save();ctx.translate(14,top+w/2);ctx.rotate(-Math.PI/2);ctx.fillText('(Z − Z₀) / L',0,0);ctx.restore();
if(!panel.data){ctx.fillStyle='#864a12';ctx.font='17px system-ui';ctx.fillText(panel.state==='unavailable'?'Result unavailable':panel.state==='error'?'Unable to load result':'Loading…',left+w/2,top+w/2);return}const data=panel.data,N=cfg.particles.length;ctx.save();ctx.beginPath();ctx.rect(left,top,w,w);ctx.clip();
function point(cycle,p){const n=(cycle*N+p)*2;return [left+(data[n]-view.x)/view.s*w,top+w-(data[n+1]-view.y)/view.s*w]}
for(let p=0;p<N;p++){const particle=cfg.particles[p];if(!selected.has(particle.id))continue;ctx.fillStyle=particle.color;for(let t=start;t<=end;t++){const [x,y]=point(t,p);ctx.fillRect(x-0.8,y-0.8,1.6,1.6)}const [sx,sy]=point(start,p);ctx.strokeStyle=particle.color;ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(sx,sy,5,0,2*Math.PI);ctx.stroke();const [x,y]=point(end,p);ctx.beginPath();ctx.arc(x,y,2.8,0,2*Math.PI);ctx.fill()}ctx.restore()}
function bindCanvas(panel){let down=null;const canvas=panel.canvas;function xy(e){const r=canvas.getBoundingClientRect();return {x:(600*(e.clientX-r.left)/r.width-53)/525,y:1-(600*(e.clientY-r.top)/r.height-18)/525}}
canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;const p=xy(e);if(p.x<0||p.x>1||p.y<0||p.y>1)return;stop();down={...p,pan:e.shiftKey,view:{...view}};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointerup',e=>{if(!down)return;const p=xy(e),d=down;down=null;if(d.pan){view.x=d.view.x+(d.x-p.x)*d.view.s;view.y=d.view.y+(d.y-p.y)*d.view.s}else{const side=Math.max(Math.abs(p.x-d.x),Math.abs(p.y-d.y));if(side<.01)return;view={x:d.view.x+Math.min(p.x,d.x)*d.view.s,y:d.view.y+Math.min(p.y,d.y)*d.view.s,s:d.view.s*side}}render()});canvas.addEventListener('pointercancel',()=>down=null)}
async function loadPanels(){stop();const serial=++loading;const rho=selectedRho();$('plots').replaceChildren();panels=[];setStatus('Loading Poincaré returns…');
for(const method of cfg.methods){const div=document.createElement('section');div.className='panel';const h=document.createElement('h2');h.textContent=method;const canvas=document.createElement('canvas');canvas.width=600;canvas.height=600;canvas.setAttribute('role','img');const note=document.createElement('p');note.className='panel-note';note.hidden=true;div.append(h,canvas,note);$('plots').append(div);const panel={canvas,note,method,data:null,state:'loading',message:''};panels.push(panel);bindCanvas(panel)}render();
await Promise.all(panels.map(async panel=>{const key=`rho-${rho}-${panel.method}`,item=cfg.cases[key];
if(item?.status==='unavailable'){panel.state='unavailable';panel.message=item.reason;panel.note.textContent=panel.message;panel.note.hidden=false;return}
try{if(!item?.asset)throw Error('Result metadata is missing.');if(!cache.has(key)){const response=await fetch(item.asset);if(!response.ok)throw Error(`Cannot load ${key}`);const bytes=await response.arrayBuffer();if(bytes.byteLength!==item.bytes)throw Error(`Incomplete data: ${key}`);const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==item.sha256)throw Error(`Data verification failed: ${key}`);if(serial!==loading)return;cache.set(key,new Float64Array(bytes))}if(serial===loading){panel.data=cache.get(key);panel.state='available'}}catch(error){panel.state='error';panel.message=error.message;panel.note.textContent=panel.message;panel.note.hidden=false}}));
if(serial!==loading)return;for(const key of cache.keys())if(!key.startsWith(`rho-${rho}-`))cache.delete(key);const available=panels.filter(p=>p.state==='available').length,failed=panels.filter(p=>p.state==='error').length;setStatus(`${cfg.cycles.toLocaleString()} cycles · ${cfg.cycles+1} saved Poincaré returns per particle, including cycle 0 · ρ = ${rho} · ${available}/${panels.length} results available${failed?` · ${failed} loading error(s)`:''}`);render()}
function animate(){if(!playing)return;const {end}=bounds();if(end>=cfg.cycles){stop();return}$('cycle').value=Math.min(cfg.cycles,end+Math.max(1,Math.ceil(cfg.cycles/200)));render();frame=requestAnimationFrame(animate)}
async function init(){const response=await fetch('data.json');if(!response.ok)throw Error('Campaign metadata is unavailable.');cfg=await response.json();$('subtitle').textContent=`${cfg.particles.length} particles · ${cfg.methods.length} fixed-step methods · ${cfg.rhos.length} gyro-radii · ${cfg.cycles} cycles`;if(cfg.publicationStatus==='partial'){$('availability').hidden=false;$('availability').textContent=`${cfg.availableCases} of ${cfg.expectedCases} results available. Missing results: ${(cfg.missingCases||[]).map(key=>{const item=cfg.cases[key];return `${item.method} at ρ = ${Number(item.rho).toFixed(2)}`}).join('; ')}. Their panels show the recorded failure.`}for(const p of cfg.particles)selected.add(p.id);for(const r of cfg.rhos){const option=document.createElement('option');option.value=r;option.textContent=Number(r).toFixed(2);$('rho').append(option)}if(cfg.rhos.includes(.3))$('rho').value=String(.3);if(!$('rho').value)$('rho').value=String(cfg.rhos[0]);for(const id of ['start','cycle'])$(id).max=cfg.cycles;$('cycle').value=cfg.cycles;
for(const p of cfg.particles){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.id=p.id;input.addEventListener('change',()=>{input.checked?selected.add(p.id):selected.delete(p.id);render()});label.style.setProperty('--particle',p.color);label.append(input,document.createTextNode(`#${p.id}`));$('particles').append(label)}
function updateSelection(ids){selected=new Set(ids);for(const input of $('particles').querySelectorAll('input'))input.checked=selected.has(Number(input.dataset.id));render()}
$('all').onclick=()=>updateSelection(cfg.particles.map(p=>p.id));$('none').onclick=()=>updateSelection([]);for(const family of cfg.families){const button=document.createElement('button');button.textContent=`Family #${family.parent}`;button.onclick=()=>updateSelection(family.ids);$('families').append(button)}
$('rho').onchange=loadPanels;for(const id of ['start','cycle']){$(id).oninput=()=>setCycle(id,Number($(id).value));$(`${id}-previous`).onclick=()=>setCycle(id,Number($(id).value)-1);$(`${id}-next`).onclick=()=>setCycle(id,Number($(id).value)+1)}$('reset-cycles').onclick=()=>{stop();$('start').value=0;$('cycle').value=cfg.cycles;render()};$('reset').onclick=()=>{view={x:0,y:0,s:1};render()};$('play').onclick=()=>{if(playing){stop();return}if(Number($('cycle').value)>=cfg.cycles)$('cycle').value=$('start').value;playing=true;$('play').textContent='Pause';animate()};await loadPanels()}
init().catch(error=>setStatus(error.message));
</script></html>'''

__all__ = ['poincare_returns', 'export_campaign']
