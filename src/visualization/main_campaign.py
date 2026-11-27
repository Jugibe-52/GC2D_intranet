"""Poincare-only static export with lazy, lossless binary assets per case."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import numpy as np

from diagnostics.io.persistence import load_solution
from studies.main_campaign import (validate_case, CampaignConfig, load_field,
	campaign_case_key, campaign_rho_label, SOURCE_SHA256)
from visualization.main_campaign_background import sample_campaign_background, validate_background_source


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


def export_campaign(directory: Path, *, allow_partial: bool = False,
		source: Path | None = None, field_grid_size: int = 128, vector_grid_size: int = 33) -> Path:
	"""Export saved returns, optionally adding a verified field without integration.

	Omitting source retains the Poincare-only publication contract. Supplying the
	original HDF5 source adds one common 51-phase field per direct-radius case.
	"""
	record: dict[str, Any] = json.loads((directory / 'campaign.json').read_text())
	missing: dict[str, Any] = {}
	if allow_partial:
		from studies.main_campaign_io import publication_cases
		_, missing = publication_cases(directory, record, allow_partial=True)
	output: Path = directory / 'site'
	assets: Path = output / 'assets'
	assets.mkdir(parents=True, exist_ok=True)
	cases: dict[str, Any] = {}
	fields: dict[str, Any] = {}
	field: Any = None
	if source is not None:
		if record['source_sha256'] != SOURCE_SHA256:
			raise ValueError('Campaign and background source fingerprints differ.')
		field = load_field(source)
	radius_units: tuple[str, float | None] | None = None
	for rho in record['rhos']:
		for method in record['methods']:
			key: str = campaign_case_key(rho, method)
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
			# Preserve the recorded convention when exporting historical campaigns.
			case_units: tuple[str, float | None] = (
				('base_cell_width', float(stored.metadata['spatial_scale_m']))
				if stored.metadata.get('schema_version', 1) == 2 else ('legacy_physical_scale', None))
			if radius_units is not None and case_units != radius_units:
				raise ValueError('Campaign cases use incompatible gyro-radius units.')
			radius_units = case_units
			if field is not None:
				validate_background_source(field, stored.metadata)
				if stored.metadata['source_sha256'] != record['source_sha256']:
					raise ValueError(f'Background source differs from saved case: {key}.')
				field_key: str = campaign_rho_label(rho)
				if field_key not in fields:
					background: dict[str, Any] = sample_campaign_background(field, stored.metadata,
						grid_size=field_grid_size, vector_grid_size=vector_grid_size)
					potential: np.ndarray = background.pop('potential')
					velocity: np.ndarray = background.pop('velocity')
					field_payload: bytes = np.concatenate((potential.ravel(), velocity.ravel())).astype('<f8').tobytes()
					field_digest: str = sha256(field_payload).hexdigest()
					field_name: str = f'field-rho-{field_key}-{field_digest[:12]}.bin'
					if len(field_payload) > 25*1024*1024:
						raise ValueError('A background exceeds the Cloudflare Pages per-file limit.')
					(assets / field_name).write_bytes(field_payload)
					fields[field_key] = dict(background, asset=f'assets/{field_name}', sha256=field_digest,
						bytes=len(field_payload), dtype='<f8', shape=list(potential.shape),
						vectorShape=list(velocity.shape), potentialValues=potential.size)
			xy: np.ndarray = poincare_returns(stored.solution, stored.metadata)
			payload: bytes = xy.tobytes()
			digest: str = sha256(payload).hexdigest()
			name: str = f'{key}-{digest[:12]}.bin'
			if len(payload) > 25*1024*1024:
				raise ValueError('A case exceeds the Cloudflare Pages per-file limit.')
			(assets / name).write_bytes(payload)
			cases[key] = dict(status='available', asset=f'assets/{name}', sha256=digest, bytes=len(payload),
				shape=list(xy.shape), rho=rho, method=method)
	if radius_units is None:
		raise ValueError('Publication requires at least one recorded gyro-radius convention.')
	radius_note = _radius_note(radius_units[0], radius_units[1], french=source is not None)
	# Compare the four Runge--Kutta recipes above the four palindromic recipes.
	display_order: tuple[str, ...] = ('RK2', 'ImplicitMidpoint', 'RK4', 'GaussLegendre4',
		'ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit')
	config: dict[str, Any] = dict(cycles=record['config']['cycles'], rhos=record['rhos'],
		methods=[name for name in display_order if name in record['methods']], particles=record['particles']['particles'],
		families=record['particles']['families'], cases=cases, spatialPeriod=2,
		rhoConvention=radius_units[0], spatialScaleM=radius_units[1],
		samplesPerCycle=1, source_sha256=record['source_sha256'], code_sha256=record['code_sha256'],
		publicationStatus='partial' if missing else 'complete', availableCases=len(cases)-len(missing),
		expectedCases=len(cases), missingCases=list(missing), fields=fields,
		rhoOptions=[dict(value=rho, label=campaign_rho_label(rho), fieldKey=campaign_rho_label(rho),
			cases={method: campaign_case_key(rho, method) for method in record['methods']}) for rho in record['rhos']])
	(output / 'data.json').write_text(json.dumps(config, separators=(',', ':'))+'\n')
	(output / 'index.html').write_text(_HTML.replace('__RHO_UNITS__', radius_note), encoding='utf-8')
	(output / '_headers').write_text('/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n/data.json\n  Cache-Control: no-cache\n')
	return output


def _radius_note(convention: str, scale: float | None, *, french: bool) -> str:
	if convention == 'base_cell_width':
		if scale is None:
			raise ValueError('The recorded base-cell width is required.')
		label = 'ρ est exprimé en largeurs de cellule ;' if french else 'ρ is measured in base-cell widths;'
		return f'{label} ρ<sub>m</sub> = ρ × L, L = {scale:.12g} m.'
	if convention == 'legacy_physical_scale':
		return 'Historical radius convention: ρ<sub>m</sub> = ρ × 0.06 / (2π) m.'
	raise ValueError(f'Unknown recorded radius convention: {convention}')


def refresh_campaign_html(output: Path) -> Path:
	"""Refresh viewer controls without rewriting saved metadata or binary assets."""
	config = json.loads((output / 'data.json').read_text())
	note = _radius_note(config['rhoConvention'], config['spatialScaleM'], french=bool(config.get('fields')))
	page = output / 'index.html'
	page.write_text(_HTML.replace('__RHO_UNITS__', note), encoding='utf-8')
	return page


_HTML = r'''<!doctype html>
<html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GC2D · Sections de Poincaré</title>
<style>
:root{font:14px Arial,sans-serif;color:#263744;background:white;--accent:#087bff}
*{box-sizing:border-box}body{margin:16px}header,main{max-width:2100px;margin:auto}
h1{font-size:22px;margin:0 0 12px}header p{line-height:1.5;margin:0 0 14px}
h2{font-size:16px;text-align:center;margin:7px 0 5px}
.bar,.playback,.actions,.families,.phase-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
label{display:inline-flex;gap:6px;align-items:center}input,select,button{font:inherit}
button,select{border:1px solid #999;border-radius:3px;padding:6px 11px;background:#f3f3f3;color:#263744}
select{background:white}button{cursor:pointer}button:hover:not(:disabled){background:#e7eff8}button:disabled{opacity:.4;cursor:default}
#families button[aria-pressed=true]{background:#dcecff;border-color:#075cbb;color:#064b97}
#families button[aria-pressed=mixed]{background:#eef5fc;border-color:#075cbb;border-style:dashed}
input[type=range],input[type=checkbox]{accent-color:var(--accent)}
.cycle-controls{display:grid;grid-template-columns:max-content 34px minmax(80px,1fr) 34px;gap:5px 8px;align-items:center;flex:1;min-width:260px}
.cycle-controls input{width:100%;min-width:0;margin:0}.cycle-controls button{padding:4px;font-size:18px;height:29px}
#interval{font-weight:bold;white-space:nowrap;font-variant-numeric:tabular-nums}
#comparison.has-field{display:grid;grid-template-columns:minmax(280px,.9fr) minmax(0,3fr);gap:24px;align-items:start}
#field-panel{min-width:0;border-right:1px solid #e2e9ed;padding-right:18px}
#field-panel[hidden]{display:none}.phase-controls{gap:7px;margin-top:12px}.phase-controls input[type=range]{flex:1;min-width:90px}
#phase-status{font-variant-numeric:tabular-nums}#field-scale{margin:0 4% 12px 9%}
.gradient{height:10px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}
.scale-values{display:flex;justify-content:space-between;font-size:12px;margin-top:3px}
#plots{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px 14px}
.panel{min-width:0}.panel-note{margin:0 10px 6px;text-align:center;color:#864a12;font-size:13px;line-height:1.4}
.panel-actions{display:flex;justify-content:center;flex-wrap:wrap;gap:6px;margin:4px 0}.panel-actions button,.panel-actions a{font-size:12px}.panel-actions a{padding:6px;color:#075cbb}.copy-status{font-size:12px;text-align:center;color:#526978;min-height:1.4em;margin:3px 0}
#availability{margin:0 0 12px;padding:9px 12px;background:#fff4de;border-left:3px solid #bd7a21;line-height:1.4}
canvas{display:block;width:100%;aspect-ratio:1;touch-action:none;cursor:crosshair}
details{margin:16px 0;padding-top:12px;border-top:1px solid #e2e9ed}summary{cursor:pointer;font-weight:bold}
#particles{display:grid;grid-template-columns:repeat(auto-fill,minmax(85px,1fr));gap:6px;margin-top:12px}
#particles label{border:1px solid #dce4e8;border-left:5px solid var(--particle);padding:6px}
.families{margin-top:12px}#status{font-size:13px;color:#526978}.legend,footer,#field-note{font-size:13px;color:#526978;line-height:1.5}footer{padding:16px 0}
@media(max-width:1100px){#comparison.has-field{grid-template-columns:1fr}#field-panel{max-width:600px;border-right:0;padding-right:0}}
@media(max-width:900px){#plots{grid-template-columns:repeat(2,minmax(0,1fr))}.playback{align-items:center}#interval{width:100%;text-align:right}}
@media(max-width:520px){body{margin:10px}#plots{grid-template-columns:1fr}.cycle-controls{order:2;flex-basis:100%}#interval{order:3}}
</style>
<header><h1>Sections de Poincaré</h1><p class="legend">Cercles vides : positions au cycle initial choisi. Petits points : retours cumulés. Grands points : retour courant. Faites glisser avec le bouton gauche pour agrandir tous les panneaux ; bouton droit + glisser ou Maj + glisser pour déplacer la vue. Vue initiale : [0,1] × [0,1]. Les coordonnées sont représentées modulo la période spatiale 2.</p></header>
<main>
<div class="bar"><label>Rayon de Larmor ρ <select id="rho" aria-label="Rayon de Larmor"></select></label><span id="subtitle">Chargement de la campagne…</span></div>
<p id="availability" hidden></p>
<div class="playback"><button id="play">Lecture</button><button id="reset-cycles">Réinitialiser les cycles</button>
<div class="cycle-controls">
<label for="start">Cycle initial</label><button id="start-previous" aria-label="Cycle initial précédent">−</button><input id="start" type="range" min="0" step="1" value="0"><button id="start-next" aria-label="Cycle initial suivant">+</button>
<label for="cycle">Cycle courant</label><button id="cycle-previous" aria-label="Cycle courant précédent">−</button><input id="cycle" type="range" min="0" step="1" value="0"><button id="cycle-next" aria-label="Cycle courant suivant">+</button>
</div><output id="interval" for="start cycle"></output></div>
<div class="actions"><button id="reset">Vue initiale [0,1]²</button><button id="period-view">Période complète [0,2]²</button><label for="point-size">Taille des points</label><input id="point-size" type="range" min="0.25" max="4" step="0.25" value="1"><output id="point-size-value" for="point-size">1×</output><button id="download-all" disabled>Télécharger les 8 évolutions (2 × 4)</button><span id="download-all-status" role="status" aria-live="polite"></span><span id="status" role="status">Chargement des retours…</span></div>
<div id="comparison">
<section id="field-panel" hidden><h2>Potentiel effectif Hρ et champ GC</h2>
<canvas id="field-canvas" width="600" height="600" role="img" aria-label="Potentiel effectif et champ de vitesse GC"></canvas>
<div id="field-scale"><div class="gradient"></div><div class="scale-values"><span id="field-min"></span><span id="field-max"></span></div><div id="field-unit"></div></div>
<div class="phase-controls"><button id="phase-play">Animer la phase</button><label for="phase">Phase</label><button id="phase-previous" aria-label="Phase précédente">−</button><input id="phase" type="range" min="0" max="50" step="1" value="0"><button id="phase-next" aria-label="Phase suivante">+</button><output id="phase-status"></output></div>
<div class="phase-controls"><label><input id="show-potential" type="checkbox" checked>Potentiel</label><label><input id="show-vectors" type="checkbox" checked>Champ de vitesse</label></div>
<p id="field-note"></p></section>
<div id="plots"></div></div>
<details><summary id="particleSummary">Particules · 84 sélectionnées</summary><div class="families"><button id="all">Tout sélectionner</button><button id="none">Tout désélectionner</button>
<button id="group-original">Groupe 1 · 40 particules</button><button id="group-chaotic">Groupe 3 · 8 candidates chaotiques</button><button id="group-periodic">Groupe 4 · 8 candidates périodiques</button>
<span id="families"></span></div><div id="particles"></div></details>
<footer>Source : PHI_2.h5 · spline cubique par réflexion · B = 1.5 · __RHO_UNITS__ <a href="data.json">Métadonnées du visualiseur</a></footer>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id), number=(x,d=4)=>Number(x).toLocaleString('fr-FR',{maximumSignificantDigits:d});
let cfg,selected=new Set(),panels=[],familyButtons=[],cache=new Map(),view={x:0,y:0,s:1},playing=false,frame=0,loading=0,paint=0,field=null,fieldError='',phaseTimer=null,savingGrid=false;
function setStatus(s){$('status').textContent=s}
// Keys come from Python metadata, never from browser rounding of small radii.
function selectedOption(){return cfg.rhoOptions[$('rho').selectedIndex]}
function selectedRho(){return selectedOption().label.replace('.',',')}
function stop(){playing=false;$('play').textContent='Lecture';cancelAnimationFrame(frame)}
function stopPhase(){clearInterval(phaseTimer);phaseTimer=null;$('phase-play').textContent='Animer la phase'}
function bounds(){const end=Math.max(0,Math.min(cfg.cycles,Math.trunc(Number($('cycle').value)||0))),start=Math.max(0,Math.min(end,Math.trunc(Number($('start').value)||0)));$('start').value=start;$('cycle').value=end;$('interval').textContent=`Cycles ${start}–${end} / ${cfg.cycles}`;for(const [id,value] of [['start',start],['cycle',end]]){$(`${id}-previous`).disabled=value===0;$(`${id}-next`).disabled=value===cfg.cycles}return {start,end}}
function setCycle(id,value){stop();$(id).value=Math.max(0,Math.min(cfg.cycles,value));if(Number($('start').value)>Number($('cycle').value))$(id==='start'?'cycle':'start').value=$(id).value;bounds();cancelAnimationFrame(paint);paint=requestAnimationFrame(render)}
function axes(c,publication=false){const ctx=c.getContext('2d'),left=publication?85:53,top=18,w=publication?480:525;ctx.clearRect(0,0,600,600);ctx.fillStyle='white';ctx.fillRect(0,0,600,600);ctx.font='12px Arial';ctx.strokeStyle='#e2e9ed';ctx.lineWidth=1;
for(let i=0;i<=4;i++){let x=left+w*i/4,y=top+w*i/4;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+w);ctx.moveTo(left,y);ctx.lineTo(left+w,y);ctx.stroke();ctx.fillStyle='#526978';ctx.textAlign='center';ctx.fillText(number(view.x+view.s*i/4,5),x,top+w+20);ctx.textAlign='right';ctx.fillText(number(view.y+view.s*(1-i/4),5),left-7,y+4)}ctx.fillStyle='#182b3a';ctx.textAlign='center';ctx.fillText('(R − R₀) / L',left+w/2,top+w+48);ctx.save();ctx.translate(14,top+w/2);ctx.rotate(-Math.PI/2);ctx.fillText('(Z − Z₀) / L',0,0);ctx.restore();return ctx}
function scheduleRender(){cancelAnimationFrame(paint);paint=requestAnimationFrame(render)}
function updateFamilyButtons(){for(const {button,ids} of familyButtons){const count=ids.filter(id=>selected.has(id)).length,all=count===ids.length;button.setAttribute('aria-pressed',all?'true':count?'mixed':'false');button.title=`${all?'Désélectionner':'Sélectionner'} la famille · ${count}/${ids.length} particules sélectionnées`}}
function render(){updateGridDownload();updateFamilyButtons();const {start,end}=bounds();$('point-size-value').textContent=`${number(Number($('point-size').value))}×`;$('particleSummary').textContent=`Particules · ${selected.size} sélectionnées`;for(const panel of panels){panel.canvas.setAttribute('aria-label',`${panel.method}, ρ = ${selectedRho()}, cycles ${start}–${end}`);panel.copy.disabled=!panel.data||panel.copying;panel.save.disabled=!panel.data||panel.saving;draw(panel,start,end)}drawField()}
function draw(panel,start,end,canvas=panel.canvas,publication=false){const ctx=axes(canvas,publication),left=publication?85:53,top=18,w=publication?480:525;if(!panel.data){ctx.fillStyle='#864a12';ctx.font='17px Arial';ctx.fillText(panel.state==='unavailable'?'Résultat indisponible':panel.state==='error'?'Échec du chargement':'Chargement…',left+w/2,top+w/2);return}const data=panel.data,N=cfg.particles.length,size=Number($('point-size').value);ctx.save();ctx.beginPath();ctx.rect(left,top,w,w);ctx.clip();
function point(cycle,p){const n=(cycle*N+p)*2;return [left+(data[n]-view.x)/view.s*w,top+w-(data[n+1]-view.y)/view.s*w]}
for(let p=0;p<N;p++){const particle=cfg.particles[p];if(!selected.has(particle.id))continue;ctx.fillStyle=particle.color;for(let t=start;t<=end;t++){const [x,y]=point(t,p);ctx.fillRect(x-.8*size,y-.8*size,1.6*size,1.6*size)}const [sx,sy]=point(start,p);ctx.strokeStyle=particle.color;ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(sx,sy,Math.max(5,2.8*size+2.2),0,2*Math.PI);ctx.stroke();const [x,y]=point(end,p);ctx.beginPath();ctx.arc(x,y,2.8*size,0,2*Math.PI);ctx.fill()}ctx.restore()}
function panelImage(panel,start,end){
// Keep the current coordinate bounds and returns; allow room for long tick labels.
const plot=document.createElement('canvas');plot.width=1200;plot.height=1200;plot.getContext('2d').scale(2,2);draw(panel,start,end,plot,true);
const image=document.createElement('canvas');image.width=1200;image.height=1320;const ctx=image.getContext('2d');ctx.scale(2,2);ctx.fillStyle='white';ctx.fillRect(0,0,600,660);ctx.textAlign='center';ctx.fillStyle='#263744';ctx.font='bold 22px Arial';ctx.fillText(panel.method,300,27);ctx.font='14px Arial';ctx.fillText(`ρ = ${selectedRho()} · Cycles ${start}–${end} · ${selected.size} particules`,300,50);ctx.drawImage(plot,0,60,600,600);return image}
function gridReady(){return panels.length===8&&panels.every(panel=>['available','unavailable','error'].includes(panel.state))&&panels.some(panel=>panel.data)}
function updateGridDownload(){$('download-all').disabled=savingGrid||!gridReady()}
function gridImage(start,end){
// Draw synchronously in display order so all eight panels share one selection snapshot.
const image=document.createElement('canvas');image.width=4800;image.height=2640;const ctx=image.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,image.width,image.height);
panels.forEach((panel,index)=>ctx.drawImage(panelImage(panel,start,end),(index%4)*1200,Math.floor(index/4)*1320));return image}
async function downloadGrid(){
if(savingGrid||!gridReady())return;stop();const {start,end}=bounds();savingGrid=true;updateGridDownload();$('download-all-status').textContent='Préparation des 8 évolutions…';
const filename=`GC2D-8-evolutions-2x4-rho-${selectedOption().label}-cycles-${start}-${end}.png`;
try{const image=gridImage(start,end);const blob=await new Promise((resolve,reject)=>image.toBlob(blob=>blob?resolve(blob):reject(Error('PNG encoding failed')),'image/png'));
const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=filename;document.body.append(link);
try{link.click();$('download-all-status').textContent='Téléchargement des 8 évolutions lancé.'}finally{link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000)}
}catch{$('download-all-status').textContent='Impossible de créer l’image. Réessayez.'}
finally{savingGrid=false;updateGridDownload()}}
function clearDownload(panel){if(panel.downloadUrl)URL.revokeObjectURL(panel.downloadUrl);panel.downloadUrl=null;panel.download.hidden=true;panel.download.removeAttribute('href')}
async function downloadPanel(panel){
if(!panel.data||panel.saving)return;stop();const {start,end}=bounds();panel.saving=true;panel.save.disabled=true;panel.copyStatus.textContent='Préparation de l’image…';
// Render the selected frame synchronously before asynchronous PNG encoding.
const filename=`GC2D-${panel.method}-rho-${selectedOption().label}-cycles-${start}-${end}.png`;
try{const image=panelImage(panel,start,end);const blob=await new Promise((resolve,reject)=>image.toBlob(blob=>blob?resolve(blob):reject(Error('PNG encoding failed')),'image/png'));
const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=filename;document.body.append(link);
try{link.click();panel.copyStatus.textContent='Téléchargement lancé.'}finally{link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000)}
}catch{panel.copyStatus.textContent='Impossible de créer l’image. Réessayez.'}
finally{panel.saving=false;panel.save.disabled=!panel.data}}
async function copyPanel(panel){
if(!panel.data||panel.copying)return;stop();const {start,end}=bounds();panel.copying=true;panel.copy.disabled=true;clearDownload(panel);panel.copyStatus.textContent='Préparation de l’image…';
const filename=`GC2D-${panel.method}-rho-${selectedOption().label}-cycles-${start}-${end}.png`;
let png;
try{
const image=panelImage(panel,start,end);png=new Promise((resolve,reject)=>image.toBlob(blob=>blob?resolve(blob):reject(Error('PNG encoding failed')),'image/png'));
// Start the clipboard write during the click, before asynchronous PNG encoding ends.
if(!navigator.clipboard?.write||typeof ClipboardItem==='undefined')throw Error('Image clipboard unavailable');
await navigator.clipboard.write([new ClipboardItem({'image/png':png})]);panel.copyStatus.textContent='Image copiée.';
}catch(error){
try{if(!png)throw error;const blob=await png;if(!panel.canvas.isConnected)return;panel.downloadUrl=URL.createObjectURL(blob);panel.download.href=panel.downloadUrl;panel.download.download=filename;panel.download.hidden=false;panel.copyStatus.textContent='Copie indisponible. Téléchargez l’image ci-dessous.'}
catch{panel.copyStatus.textContent='Impossible de créer l’image. Réessayez.'}
}finally{panel.copying=false;panel.copy.disabled=!panel.data}}
function color(t){const stops=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],v=Math.max(0,Math.min(1,t))*4,i=Math.min(3,Math.floor(v)),a=v-i;return stops[i].map((c,k)=>Math.round(c*(1-a)+stops[i+1][k]*a))}
function drawField(){if($('field-panel').hidden)return;const ctx=axes($('field-canvas')),phase=Number($('phase').value),left=53,top=18,w=525;
$('phase-status').textContent=`${phase} / 50 · ${number(phase/50)} cycle`;$('phase-previous').disabled=phase===0;$('phase-next').disabled=phase===50;
if(!field){$('field-scale').hidden=true;ctx.fillStyle='#864a12';ctx.fillText(fieldError?'Fond indisponible':'Chargement…',left+w/2,top+w/2);$('field-note').textContent=fieldError;return}
const b=field.bounds,n=field.shape[1],vn=field.vectorShape[1],px=x=>left+(x-view.x)/view.s*w,py=y=>top+w-(y-view.y)/view.s*w;
ctx.save();ctx.beginPath();ctx.rect(left,top,w,w);ctx.clip();
if($('show-potential').checked){if(!field.images.has(phase)){const raster=document.createElement('canvas');raster.width=n;raster.height=n;const rc=raster.getContext('2d'),im=rc.createImageData(n,n),range=field.maximum-field.minimum||1;for(let y=0;y<n;y++)for(let x=0;x<n;x++){const rgb=color((field.values[(phase*n+y)*n+x]-field.minimum)/range),k=((n-1-y)*n+x)*4;im.data.set([...rgb,255],k)}rc.putImageData(im,0,0);field.images.set(phase,raster)}ctx.imageSmoothingEnabled=false;ctx.drawImage(field.images.get(phase),px(b.x),py(b.y+b.span),b.span/view.s*w,b.span/view.s*w)}
if($('show-vectors').checked&&field.maxSpeed>0){const scale=.75*b.span/vn/field.maxSpeed;ctx.strokeStyle=$('show-potential').checked?'#fff':'#263744';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1.1;
for(let y=0;y<vn;y++)for(let x=0;x<vn;x++){const i=field.potentialValues+((phase*vn+y)*vn+x)*2,ux=field.values[i],uy=field.values[i+1];if(Math.hypot(ux,uy)<1e-12*field.maxSpeed)continue;const sx=px(b.x+(x+.5)*b.span/vn),sy=py(b.y+(y+.5)*b.span/vn),dx=ux*scale/view.s*w,dy=-uy*scale/view.s*w,ex=sx+dx,ey=sy+dy,angle=Math.atan2(dy,dx),head=Math.min(4,Math.hypot(dx,dy)*.4);ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ex,ey);ctx.stroke();ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-head*Math.cos(angle-.5),ey-head*Math.sin(angle-.5));ctx.lineTo(ex-head*Math.cos(angle+.5),ey-head*Math.sin(angle+.5));ctx.closePath();ctx.fill()}}
ctx.restore();$('field-scale').hidden=!$('show-potential').checked;$('field-min').textContent=number(field.minimum);$('field-max').textContent=number(field.maximum);$('field-unit').textContent=`Hρ · ${field.units}`;$('field-note').textContent=`Hρ = Φρ / B ; f = (−∂yHρ, ∂xHρ). Échelles fixes sur les 51 phases ; vitesse maximale ${number(field.maxSpeed)} ${field.velocityUnits}. Les phases 0 et 1 coïncident après vérification. Le potentiel et le champ suivent les réflexions réelles.`}
function bindCanvas(canvas){let down=null;function xy(e){const r=canvas.getBoundingClientRect();return {x:(600*(e.clientX-r.left)/r.width-53)/525,y:1-(600*(e.clientY-r.top)/r.height-18)/525}}
function clampView(){view.x=Math.max(0,Math.min(2-view.s,view.x));view.y=Math.max(0,Math.min(2-view.s,view.y))}
function pan(p){view={x:down.view.x+(down.x-p.x)*down.view.s,y:down.view.y+(down.y-p.y)*down.view.s,s:down.view.s};clampView()}
function clearDrag(){const id=down?.id;down=null;canvas.style.cursor='';if(id!==undefined&&canvas.hasPointerCapture(id))canvas.releasePointerCapture(id)}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{if(down||(e.button!==0&&e.button!==2))return;const p=xy(e);if(p.x<0||p.x>1||p.y<0||p.y>1)return;e.preventDefault();stop();down={...p,id:e.pointerId,pan:e.button===2||e.shiftKey,view:{...view}};if(down.pan)canvas.style.cursor='grabbing';canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!down||e.pointerId!==down.id||!down.pan)return;e.preventDefault();pan(xy(e));scheduleRender()});
canvas.addEventListener('pointerup',e=>{if(!down||e.pointerId!==down.id)return;const p=xy(e),d=down;if(d.pan){pan(p)}else{const side=Math.max(Math.abs(p.x-d.x),Math.abs(p.y-d.y));if(side<.01){clearDrag();return}view={x:d.view.x+Math.min(p.x,d.x)*d.view.s,y:d.view.y+Math.min(p.y,d.y)*d.view.s,s:Math.min(2,d.view.s*side)};clampView()}clearDrag();scheduleRender()});
for(const type of ['pointercancel','lostpointercapture'])canvas.addEventListener(type,e=>{if(down&&e.pointerId===down.id)clearDrag()})}
async function verifiedAsset(item,key){const response=await fetch(item.asset);if(!response.ok)throw Error(`Chargement impossible : ${key}`);const bytes=await response.arrayBuffer();if(bytes.byteLength!==item.bytes)throw Error(`Données incomplètes : ${key}`);const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==item.sha256)throw Error(`Vérification des données échouée : ${key}`);return new Float64Array(bytes)}
function failureText(item){return /converg/i.test(item.reason||'')?'Le solveur non linéaire n’a pas convergé.':'La simulation n’a pas produit de résultat complet.'}
async function loadField(option,serial){field=null;fieldError='';const item=cfg.fields?.[option.fieldKey];if(!Object.keys(cfg.fields||{}).length)return;if(!item){fieldError='Aucun fond vérifié pour ce rayon.';return}try{const values=await verifiedAsset(item,`Hρ, ρ = ${option.label}`);if(serial===loading)field={...item,values,images:new Map()}}catch(error){if(serial===loading)fieldError=error.message}}
async function loadPanels(){stop();stopPhase();const serial=++loading,option=selectedOption();for(const panel of panels)clearDownload(panel);$('plots').replaceChildren();panels=[];field=null;fieldError='';setStatus('Chargement des retours…');
for(const method of cfg.methods){const div=document.createElement('section');div.className='panel';const h=document.createElement('h2');h.textContent=method;const canvas=document.createElement('canvas');canvas.width=600;canvas.height=600;canvas.setAttribute('role','img');const note=document.createElement('p');note.className='panel-note';note.hidden=true;
const actions=document.createElement('div');actions.className='panel-actions';const save=document.createElement('button');save.type='button';save.textContent='Télécharger l’image';save.setAttribute('aria-label',`Télécharger l’image ${method}`);save.disabled=true;const copy=document.createElement('button');copy.type='button';copy.textContent='Copier l’image';copy.setAttribute('aria-label',`Copier l’image ${method}`);copy.disabled=true;const download=document.createElement('a');download.textContent='Télécharger l’image';download.hidden=true;download.setAttribute('aria-label',`Télécharger la copie ${method}`);actions.append(save,copy,download);const copyStatus=document.createElement('p');copyStatus.className='copy-status';copyStatus.setAttribute('role','status');div.append(h,canvas,note,copyStatus,actions);$('plots').append(div);const panel={canvas,note,method,save,copy,download,copyStatus,saving:false,copying:false,downloadUrl:null,data:null,state:'loading',message:''};save.onclick=()=>downloadPanel(panel);copy.onclick=()=>copyPanel(panel);panels.push(panel);bindCanvas(canvas)}render();
await Promise.all([loadField(option,serial),...panels.map(async panel=>{const key=option.cases[panel.method],item=cfg.cases[key];if(item?.status==='unavailable'){panel.state='unavailable';panel.message=failureText(item);panel.note.textContent=panel.message;panel.note.hidden=false;return}try{if(!item?.asset)throw Error('Métadonnées du résultat absentes.');if(!cache.has(key)){const data=await verifiedAsset(item,key);if(serial!==loading)return;cache.set(key,data)}if(serial===loading){panel.data=cache.get(key);panel.state='available'}}catch(error){if(serial===loading){panel.state='error';panel.message=error.message;panel.note.textContent=panel.message;panel.note.hidden=false}}})]);
if(serial!==loading)return;const currentKeys=new Set(Object.values(option.cases));for(const key of cache.keys())if(!currentKeys.has(key))cache.delete(key);const available=panels.filter(p=>p.state==='available').length,failed=panels.filter(p=>p.state==='error').length;setStatus(`${cfg.cycles.toLocaleString('fr-FR')} cycles · ${cfg.cycles+1} retours par particule, cycle 0 inclus · ρ = ${selectedRho()} · ${available}/${panels.length} résultats disponibles${failed?` · ${failed} erreur(s) de chargement`:''}${fieldError?' · fond indisponible':''}`);render()}
function animate(){if(!playing)return;const {end}=bounds();if(end>=cfg.cycles){stop();return}$('cycle').value=Math.min(cfg.cycles,end+Math.max(1,Math.ceil(cfg.cycles/200)));render();frame=requestAnimationFrame(animate)}
async function init(){const response=await fetch('data.json');if(!response.ok)throw Error('Métadonnées de la campagne indisponibles.');cfg=await response.json();$('subtitle').textContent=`${cfg.particles.length} particules · ${cfg.methods.length} méthodes à pas fixe · ${cfg.rhos.length} rayons · ${cfg.cycles} cycles`;if(cfg.publicationStatus==='partial'){$('availability').hidden=false;$('availability').textContent=`${cfg.availableCases} résultats disponibles sur ${cfg.expectedCases}. Résultats manquants : ${(cfg.missingCases||[]).map(key=>{const item=cfg.cases[key],option=cfg.rhoOptions.find(o=>o.value===item.rho);return `${item.method}, ρ = ${(option?.label||String(item.rho)).replace('.',',')}`}).join(' ; ')}. Les panneaux correspondants restent visibles.`}
for(const p of cfg.particles)selected.add(p.id);for(const r of cfg.rhoOptions){const option=document.createElement('option');option.value=r.fieldKey;option.textContent=r.label.replace('.',',');$('rho').append(option)}const preferred=cfg.rhoOptions.findIndex(o=>o.value===.015);if(preferred>=0)$('rho').selectedIndex=preferred;for(const id of ['start','cycle'])$(id).max=cfg.cycles;$('cycle').value=cfg.cycles;
const withField=Object.keys(cfg.fields||{}).length>0;$('field-panel').hidden=!withField;$('comparison').classList.toggle('has-field',withField);bindCanvas($('field-canvas'));
for(const p of cfg.particles){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.id=p.id;input.addEventListener('change',()=>{input.checked?selected.add(p.id):selected.delete(p.id);render()});label.style.setProperty('--particle',p.color);label.append(input,document.createTextNode(`#${p.id}`));$('particles').append(label)}
function updateSelection(ids){selected=new Set(ids);for(const input of $('particles').querySelectorAll('input'))input.checked=selected.has(Number(input.dataset.id));render()}
// Chapter 4 subsets retain the IDs in data/particles/{original_40,chaotic_candidates,periodic_candidates_8}.json.
$('group-original').onclick=()=>updateSelection(cfg.particles.filter(p=>p.id<=40).map(p=>p.id));
$('group-chaotic').onclick=()=>updateSelection([12,15,18,21,24,27,30,33]);
$('group-periodic').onclick=()=>updateSelection([1,2,60,64,65,72,76,83]);
$('all').onclick=()=>updateSelection(cfg.particles.map(p=>p.id));$('none').onclick=()=>updateSelection([]);for(const family of cfg.families){const button=document.createElement('button');button.textContent=`Famille #${family.parent}`;button.onclick=()=>{const remove=family.ids.every(id=>selected.has(id));for(const id of family.ids)remove?selected.delete(id):selected.add(id);updateSelection(selected)};familyButtons.push({button,ids:family.ids});$('families').append(button)}
$('download-all').onclick=downloadGrid;
$('point-size').oninput=scheduleRender;
$('rho').onchange=loadPanels;for(const id of ['start','cycle']){$(id).oninput=()=>setCycle(id,Number($(id).value));$(`${id}-previous`).onclick=()=>setCycle(id,Number($(id).value)-1);$(`${id}-next`).onclick=()=>setCycle(id,Number($(id).value)+1)}$('reset-cycles').onclick=()=>{stop();$('start').value=0;$('cycle').value=cfg.cycles;render()};$('reset').onclick=()=>{view={x:0,y:0,s:1};render()};$('period-view').onclick=()=>{view={x:0,y:0,s:2};render()};$('play').onclick=()=>{if(playing){stop();return}if(Number($('cycle').value)>=cfg.cycles)$('cycle').value=$('start').value;playing=true;$('play').textContent='Pause';animate()};
function setPhase(value){stopPhase();$('phase').value=Math.max(0,Math.min(50,value));drawField()}
$('phase').oninput=()=>setPhase(Number($('phase').value));$('phase-previous').onclick=()=>setPhase(Number($('phase').value)-1);$('phase-next').onclick=()=>setPhase(Number($('phase').value)+1);$('phase-play').onclick=()=>{if(phaseTimer){stopPhase();return}$('phase-play').textContent='Pause phase';phaseTimer=setInterval(()=>{$('phase').value=(Number($('phase').value)+1)%51;drawField()},150)};for(const id of ['show-potential','show-vectors'])$(id).onchange=drawField;await loadPanels()}
init().catch(error=>setStatus(error.message));
</script></html>
'''

__all__ = ['poincare_returns', 'export_campaign', 'refresh_campaign_html']
