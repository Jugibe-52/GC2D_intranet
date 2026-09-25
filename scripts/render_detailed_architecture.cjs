/** Render a detailed diagram from its canonical PlantUML/DOT source.
 * Requires @viz-js/viz and sharp, resolved normally or through NODE_PATH.
 * Optional argument: a .puml path; defaults to the detailed package diagram.
 * Only the SVG and PNG matching the selected source are written.
 */
const fs = require('node:fs/promises');
const path = require('node:path');
const { instance } = require('@viz-js/viz');
const sharp = require('sharp');

async function main() {
  if (process.argv.length > 3) throw new Error('Usage: node scripts/render_detailed_architecture.cjs [diagram.puml]');
  const sourcePath = process.argv[2]
    ? path.resolve(process.argv[2])
    : path.resolve(__dirname, '../docs/simulation/package-integration-architecture.puml');
  if (path.extname(sourcePath) !== '.puml') throw new Error('The source must be a .puml file.');
  const stem = sourcePath.slice(0, -5);
  const source = await fs.readFile(sourcePath, 'utf8');
  const match = source.match(/^@startdot\s*\n([\s\S]*?)\n@enddot\s*$/);
  if (!match) throw new Error('Expected one complete @startdot diagram.');

  const viz = await instance();
  const result = viz.render(match[1], { engine: 'dot', format: 'svg' });
  for (const error of result.errors) console.error(`${error.level}: ${error.message}`);
  if (result.status !== 'success') throw new Error('Graphviz could not render the architecture.');
  const svg = result.output.split('\n').map(line => line.trimEnd()).join('\n');
  // Rasterize the same SVG, so cards, labels and arrows agree in both formats.
  const png = await sharp(Buffer.from(svg), { density: 110 }).png().toBuffer();
  await fs.writeFile(`${stem}.svg`, svg);
  await fs.writeFile(`${stem}.png`, png);
  const { width, height } = await sharp(png).metadata();
  console.log(`Rendered ${path.basename(stem)}.svg and .png (${width} x ${height}).`);
}

main().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
