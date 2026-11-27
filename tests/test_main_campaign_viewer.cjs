// Run with: node --test tests/test_main_campaign_viewer.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/visualization/main_campaign.py'), 'utf8');
const script = source.split('<script>')[1].split('</script>')[0]
  .replace('init().catch(error=>setStatus(error.message));', '');

function canvas(width = 600) {
  const operations = [], listeners = new Map(), captures = new Set();
  const ctx = new Proxy({}, {get: (target, name) => target[name] ||
    ((...args) => operations.push([name, ...args]))});
  return {
    width: 600, height: 600, style: {}, operations,
    getContext: () => ctx,
    getBoundingClientRect: () => ({left: 0, top: 0, width, height: width}),
    setAttribute() {},
    addEventListener: (name, callback) => listeners.set(name, callback),
    setPointerCapture: id => captures.add(id),
    hasPointerCapture: id => captures.has(id),
    releasePointerCapture: id => captures.delete(id),
    send(type, x, y, extra = {}) {
      const event = {pointerId: 1, button: 2, shiftKey: false,
        clientX: (53 + 525*x)*width/600, clientY: (18 + 525*(1-y))*width/600,
        preventDefault() { this.prevented = true; }, ...extra};
      listeners.get(type)?.(event);
      return event;
    }
  };
}

function viewer(width) {
  const elements = new Map(), images = [], links = [], encoders = [], revoked = [], timers = [], target = canvas(width);
  const element = id => {
    if (!elements.has(id)) elements.set(id, {value: '0', textContent: '', hidden: true});
    return elements.get(id);
  };
  element('point-size').value = '1';
  let pending;
  const sandbox = vm.createContext({document: {getElementById: element,
    body: {append() {}},
    createElement: tag => {
      if (tag === 'a') { const link = {click() {this.clicked = true;}, remove() {this.removed = true;}}; links.push(link); return link; }
      const image = canvas(); image.toBlob = callback => encoders.push(callback); images.push(image); return image;
    }},
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL: url => revoked.push(url)},
    setTimeout: callback => timers.push(callback),
    cancelAnimationFrame() { pending = undefined; },
    requestAnimationFrame(callback) { pending = callback; return 1; },
    clearInterval() {}, target});
  const run = code => vm.runInContext(code, sandbox);
  run(script);
  run(`cfg={methods:Array.from({length:8},(_,i)=>'method'+i),cycles:2,particles:[{id:1,color:'#ff0000'}],rhoOptions:[{label:'0.015'}]};
    $('rho').selectedIndex=0; selected.add(1);
    panels=[{canvas:target,copy:{},save:{},method:'RK4',data:new Float64Array([.2,.2,.3,.3,.4,.4])}];
    bindCanvas(target);`);
  return {target, images, links, encoders, revoked, timers, element, run,
    view: () => JSON.parse(run('JSON.stringify(view)')),
    flush: () => { const callback = pending; pending = undefined; callback?.(); }};
}

function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);
}

test('right drag updates the shared view before release on a resized canvas', () => {
  const v = viewer(300);
  assert.ok(v.target.send('pointerdown', .6, .6).prevented);
  assert.equal(v.target.style.cursor, 'grabbing');
  v.target.send('pointermove', .4, .3);
  near(v.view().x, .2); near(v.view().y, .3); near(v.view().s, 1);
  v.flush();
  assert.ok(v.target.operations.some(([name]) => name === 'fillText'));
  v.target.send('pointerup', .3, .2);
  near(v.view().x, .3); near(v.view().y, .4);
  assert.equal(v.target.style.cursor, '');
  assert.equal(v.target.hasPointerCapture(1), false);
  assert.ok(v.target.send('contextmenu', 0, 0).prevented);
});

test('pan stays within the period-two bounds, including release outside the canvas', () => {
  const v = viewer();
  v.target.send('pointerdown', .5, .5);
  v.target.send('pointermove', -5, -5);
  assert.deepEqual(v.view(), {x: 1, y: 1, s: 1});
  v.target.send('pointerup', 5, 5);
  assert.deepEqual(v.view(), {x: 0, y: 0, s: 1});
});

test('left-drag zoom and Shift-left-drag pan remain available', () => {
  const v = viewer();
  v.target.send('pointerdown', .2, .2, {button: 0});
  v.target.send('pointerup', .6, .6, {button: 0});
  near(v.view().x, .2); near(v.view().y, .2); near(v.view().s, .4);
  v.target.send('pointerdown', .5, .5, {button: 0, shiftKey: true});
  v.target.send('pointermove', .25, .25, {button: -1});
  near(v.view().x, .3); near(v.view().y, .3); near(v.view().s, .4);
  v.target.send('pointerup', .25, .25, {button: 0});
});

test('unrelated pointers, middle clicks and clicks outside the plot do not move it', () => {
  const v = viewer();
  v.target.send('pointerdown', .5, .5, {button: 1});
  v.target.send('pointermove', .1, .1);
  v.target.send('pointerdown', -.1, .5);
  v.target.send('pointermove', .1, .1);
  v.target.send('pointerdown', .5, .5);
  v.target.send('pointermove', .1, .1, {pointerId: 2});
  v.target.send('pointerup', .1, .1, {pointerId: 2});
  assert.deepEqual(v.view(), {x: 0, y: 0, s: 1});
});

for (const type of ['pointercancel', 'lostpointercapture']) {
  test(`${type} ends dragging and allows the next gesture`, () => {
    const v = viewer();
    v.target.send('pointerdown', .5, .5);
    v.target.send(type, .5, .5);
    v.target.send('pointermove', .1, .1);
    assert.deepEqual(v.view(), {x: 0, y: 0, s: 1});
    assert.equal(v.target.style.cursor, '');
    v.target.send('pointerdown', .5, .5);
    v.target.send('pointerup', .25, .25);
    near(v.view().x, .25);
  });
}

test('point sizes scale accumulated and current returns on screen and in PNG rendering', () => {
  const v = viewer();
  for (const size of [.25, 1, 2, 4]) {
    v.element('point-size').value = String(size);
    v.target.operations.length = 0;
    v.run('draw(panels[0],0,2)');
    const rects = v.target.operations.filter(([name, x, y, w]) => name === 'fillRect' && w !== 600);
    assert.equal(rects.length, 3);
    rects.forEach(([, x, y, w, h]) => { near(w, 1.6*size); near(h, w); });
    const arcs = v.target.operations.filter(([name]) => name === 'arc');
    near(arcs[1][3], 2.8*size);
    assert.ok(arcs[0][3] >= arcs[1][3] + 2.2 - 1e-12);
    v.run('panelImage(panels[0],0,2)');
    const plot = v.images.at(-2), image = v.images.at(-1);
    assert.equal(image.width, 1200); assert.equal(image.height, 1320);
    assert.deepEqual(plot.operations[0], ['scale', 2, 2]);
    const exported = plot.operations.filter(([name, x, y, w]) => name === 'fillRect' && w !== 600);
    exported.forEach(([, x, y, w]) => near(w, 1.6*size));
    near(plot.operations.filter(([name]) => name === 'arc')[1][3], 2.8*size);
  }
});

function eightPanels(v) {
  v.run("panels=cfg.methods.map(method=>({...panels[0],method,state:'available'}))");
}

test('combined PNG has two rows of four ordered panels, including explicit missing and load-error slots', () => {
  const v = viewer(); eightPanels(v);
  v.run("panels[1].data=null;panels[1].state='unavailable';panels[5].data=null;panels[5].state='error';view={x:.1,y:.2,s:.5}");
  v.element('point-size').value = '2';
  const grid = v.run('gridImage(1,2)');
  assert.equal(grid.width, 4800); assert.equal(grid.height, 2640);
  const cells = grid.operations.filter(([op]) => op === 'drawImage');
  assert.equal(cells.length, 8);
  cells.forEach(([,cell,x,y], i) => {
    assert.equal(x, (i%4)*1200); assert.equal(y, Math.floor(i/4)*1320);
    assert.ok(cell.operations.some(([op,text]) => op === 'fillText' && text === 'method'+i));
    assert.ok(cell.operations.some(([op,text]) => op === 'fillText' && text === 'ρ = 0,015 · Cycles 1–2 · 1 particules'));
    const plot = cell.operations.find(([op]) => op === 'drawImage')[1];
    if (i === 1 || i === 5) {
      assert.ok(plot.operations.some(([op,text]) => op === 'fillText' && text === (i===1?'Résultat indisponible':'Échec du chargement')));
    } else {
      const returns = plot.operations.filter(([op,x,y,w]) => op === 'fillRect' && w !== 600);
      assert.equal(returns.length, 2);
      returns.forEach(([,x,y,w]) => near(w, 3.2));
    }
  });
});

test('grid download is disabled during loading or with no available data', () => {
  const v = viewer(); eightPanels(v);
  assert.equal(v.run('gridReady()'), true);
  v.run("panels[2].state='loading';updateGridDownload()");
  assert.equal(v.element('download-all').disabled, true);
  v.run('downloadGrid()'); assert.equal(v.images.length, 0);
  v.run("panels.forEach(p=>{p.data=null;p.state='unavailable'});updateGridDownload()");
  assert.equal(v.element('download-all').disabled, true);
});

test('download captures one selection before encoding and blocks duplicate clicks', async () => {
  const v = viewer(); eightPanels(v); v.element('cycle').value='2';
  v.run('playing=true');
  const pending = v.run('downloadGrid()');
  assert.equal(v.run('playing'), false);
  assert.equal(v.element('download-all').disabled, true);
  assert.equal(v.encoders.length, 1);
  await v.run('downloadGrid()'); assert.equal(v.encoders.length, 1);
  v.run("cfg.rhoOptions[0].label='0.025';selected.clear();panels=[];updateGridDownload()");
  v.encoders[0]({}); await pending;
  assert.equal(v.links[0].download, 'GC2D-8-evolutions-2x4-rho-0.015-cycles-0-2.png');
  assert.equal(v.links[0].clicked, true); assert.equal(v.links[0].removed, true);
  assert.equal(v.element('download-all').disabled, true);
  assert.match(v.element('download-all-status').textContent, /lancé/);
  v.timers[0](); assert.deepEqual(v.revoked, ['blob:test']);
});

test('PNG encoding failure reports failure and restores the button for retry', async () => {
  const v = viewer(); eightPanels(v);
  const pending = v.run('downloadGrid()');v.encoders[0](null);await pending;
  assert.equal(v.links.length, 0);assert.equal(v.element('download-all').disabled, false);
  assert.match(v.element('download-all-status').textContent, /Impossible/);
  assert.equal(v.run('savingGrid'), false);
});
