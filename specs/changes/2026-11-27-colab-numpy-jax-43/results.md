# Execution record

Accepted and authorized on 2026-11-27: adapt and publish both section 4.3 notebooks using the established Colab/public-B2/personal-Modal workflow, with real links in the French thesis. Both notebooks and their versioned public inputs are now published.

The visualization downloads the saved reference without credentials or integrations. The calculation guides personal Modal authentication, checks frozen inputs before loading them, runs paired backends with compilation-cache audits, recovers confirmed calls and offers a checked result ZIP for download. The separate visualization shows the thesis reference; it does not import a reader's new result ZIP.

The private scientific archive, numerical source modules and original 72-integration reference remain unchanged. The exact previous calculation notebook is retained at `outputs/colab-backups/04_03_numpy_jax/calcul-before-colab.ipynb`. Historical calculation and visualization notebooks remain in `outputs/studies/bm4_numpy_jax/methods-20261119-union16-rho0015-c10-30-90/reproduction-union16.tar.gz`; this archive predates the section-folder reorganization and is not an exact pre-Colab snapshot of the viewer.

## Scientific and runtime checks

- Verified all nine original cases, 72 ordered timing measurements, 18 trajectory manifests, float64 states, aligned sample times, initial conditions, particle IDs and group maxima. The public viewer recomputes a summary exactly equal to the original figure data using unwrapped Euclidean distances. First JAX calls record one compilation-cache miss; later calls record one hit. Repeated original trajectories were not retained, so their bitwise identity remains the original worker's recorded audit.
- The five focused visualization tests passed. All six code cells also completed locally with `validate_visualisation.py --live --save-outputs`, using anonymous public HTTPS and the unchanged downloader. Both PNG figures are stored in the notebook, whose metadata records `execution_mode=public_https`.
- The public kit preserves all 339 source members byte for byte while clearing filesystem ownership metadata from archive headers. Only `potential.metadata.source_path` is normalized to `/data/potential/V1/PHI_2.h5` in the nine serialized inputs. Original and public hashes remain linked; comparison of object fields and numerical arrays after decoding, plus restoration and re-encoding of the original path, verified that numerical inputs were preserved. No credentials or personal account paths remain in the public notebook or kit.
- A local one-cycle smoke test used all three methods, one first call and three later calls per backend: 24 integrations and six saved trajectories. The final ZIP is `outputs/colab-validation/04_03_numpy_jax/local-smoke-final/validation-one-cycle.zip`. Independent verification accepted it and rejected altered cache records, phases, states, nonfinite process times and settings even after recomputing the outer checksums. Notebook structure and Python cells passed validation.
- In-memory recovery checks confirmed reuse of a saved call and rejection of ambiguous submissions, changed settings and conflicting call IDs. Recovery with an explicitly supplied known call ID passed without submitting a job.
- `latexmk -pdf main_fr.tex` completed successfully after adding both links to French section 4.3. The PDF has 80 pages; its five existing underfull-box messages are unchanged, with no new warnings, errors or undefined references. The PDF was not opened or rendered.

## Publication checks

Both Drive files inherit reader access from their destination folder. Their Colab pages display the expected cells and reciprocal notebook links. The shared personal publication registry and configuration record are updated, with destination and remote-listing checks passing. Published notebook bytes correspond to these SHA-256 digests:

| Notebook | Colab link | SHA-256 |
| --- | --- | --- |
| Visualization | [Open](https://colab.research.google.com/drive/1xVRKtB2AWp_Pqv6COtP1F3z0Ystfa2Bo) | `4ae17b906fb20700cfe314d6459eaf72fb0d28c8adb9c94e2cc9677611d11a35` |
| Calculation | [Open](https://colab.research.google.com/drive/1X-iiPUOFq35JgXlZF9XhMJQwZ-8DOBQ4) | `952adb1972a78ab420dbeffa2c9b7f30137fe035d4e68d4ad669c12951d2d160` |

The public visualization bundle is versioned under `gc2d-colab-public/04_03_numpy_jax/methods-20261119-union16-rho0015-c10-30-90/v1`; its pinned manifest SHA-256 is `327a81f8566269eee2976322ec6083fe584c4d7e001b9401a8768f23547de398`. Anonymous downloads verified every object, all 72 measurements and all 18 saved trajectories.

The calculation objects are under `gc2d-colab-public/04_03_numpy_jax/calculation/v1`. Anonymous HTTPS returned HTTP 200 and bytes identical to the local files: `manifest.json` has 413 bytes and SHA-256 `6199970737d76410f6e72916c2505a4fe4f7ea7edba34343552f0305120ac73b`; `calculation-kit.tar.xz` has 3,809,900 bytes and SHA-256 `33c91014d7ed2213bed2a8565f72822c51c02cc714c0d05304ae8ac7997ff446`. `prepare_kit` accepted the pinned kit and checked its 20 declared input files, nine payloads and 339 source members.

These checks do not include an actual Google Colab runtime or a new remote section-4.3 Modal execution. The reduced local calculation is a workflow check, not a replacement reference benchmark. `.venv/bin/python scripts/check_sdd.py` passed for 314 records and 22 requirements, and the changed tracked documentation passed `git diff --check`.
