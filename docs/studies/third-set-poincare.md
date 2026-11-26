# Third-set Poincare section

Open [visualisation_troisieme_ensemble.ipynb](../../../TFM_redaccion/notebooks/archives_ensembles_particules/visualisation_troisieme_ensemble.ipynb) in the existing GC2D Python environment.
The notebook reads the sibling thesis's [reference selection](../../../TFM_redaccion/datos/campana_referencia.json),
currently `main-20261117-rho-div20` with representative `rho=0.015`, and extracts
the saved BM4Implicit result. It launches no trajectory integration.
The eight particles in `data/particles/chaotic_candidates.json` retain IDs
12, 15, 18, 21, 24, 27, 30 and 33, their exact initial positions and original colors.
The source campaign and its [website](https://gc2d-main-reflected-rho-div20.pages.dev/)
are the current thesis reference; earlier campaigns remain historical.

## Saved numerical source

The source case `rho-0.015-BM4Implicit` contains all 84 particles, 5000 forcing
cycles, 40 fixed steps per cycle and ten saved samples per cycle. The extraction
selects the eight particle IDs and every tenth sample, including cycle zero.
The radius is measured in base-cell widths and reaches gyroaveraging unchanged:
`rho_runtime == rho`. Its physical equivalent `rho_m = rho * spatial_scale_m`
is provenance only: rho=0.015 represents 0.0027 metres for a 0.18-metre cell.
The factor 20 belongs to the choice of input radii, not a conversion in the code.

The measured potential uses constant field 0 and variable field 15 from
`PHI_2.h5`, cubic interpolation, no smoothing and the reflected spatial extension.
Space is normalized by the base-cell width and time by the forcing period.
The magnetic field is B=1.5 T; the spatial extension has period two.
The recorded Newton controls are absolute tolerance 1e-12, relative tolerance
1e-11 and at most 40 corrections with analytic Jacobians. The saved calculation
used Modal JAX CPU float64. Every method records the same direct H, analytic
partial_t H and H-H0 diagnostics, without energy quadrature.

Before extraction, the notebook checks source-file sizes and SHA256 hashes,
archive schemas, the selected campaign configuration, radius convention,
source-field fingerprint, exact particle records, finite float64 states,
aligned times, 200000 fixed steps and zero output interpolation.
Use canonical campaign keys to retain all radius digits; formatting rho to two
decimals would select the wrong case for rho=0.015.

## Plotting and exports

The reference selection supplies the campaign and representative radius.
`first_cycle`, `last_cycle`, `view_limits` and `show_legend` control the plot;
`export_thesis_figure` controls copying to the existing sibling thesis directory.
Changing a plotting parameter only reads and extracts saved results.
The complete campaign is prepared or recovered through the section 4.2
[calculation notebook](../../notebooks/04_02_execution_principale/calcul.ipynb);
its visualization is provided by the campaign website.

Derived NPZ returns, JSON provenance and French-labelled PNG/PDF figures belong
under `notebooks/developements/main_reflected_campaign/main-20261117-rho-div20/postprocessing/chapter4/current/`.
The extracted return archive contains 5001 time rows and 16 state components.
The figure displays cycles 1–5000 by default; the saved initial row is excluded
from the plotted cloud. Source archive, metadata and particle-dataset hashes
identify the numerical data behind each figure.

Coordinates remain unwrapped. The default viewport is `[0,1]²`; it is a viewing
window, not a dynamical boundary. Returns outside it are counted and retained.
The campaign website displays coordinates modulo two, so its viewport membership
need not agree with the unwrapped thesis figure. The plot preserves one original
color per particle; `show_legend` enables a French particle legend.

Thesis exports are `TFM_redaccion/figuras/tercer_conjunto_poincare_fr.png`, the
corresponding PDF and JSON provenance. This plot does not verify chaos,
confinement or long-time numerical convergence. Neither extraction nor plotting
modifies a solver or potential implementation. See the
[main reflected campaign](main-reflected-campaign.md) for the full protocol,
availability and storage records.

## Reference counts and historical outputs

The saved rho=0.015 BM4Implicit result supplies 40000 non-initial returns for
these eight particles: 36836 lie inside the default viewport and 3164 outside.
IDs 27 and 33 account for 305 and 2859 outside returns respectively.
These counts concern unwrapped positions over cycles 1–5000; they do not
establish the particles' dynamical classification.

The earlier standalone archive
`data/trajectory/third_set_poincare/bm4implicit_2e069839791cac95.npz` and the
`main-20261116-direct-rho` extraction at rho=0.30 remain historical products
with their original radius conventions and provenance. Neither is a fallback
for missing results of the current reference campaign.
