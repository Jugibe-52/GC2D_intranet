# Execution record

Accepted and authorized on 2026-11-27 after a read-only comparison: clean section-4.4 notebooks locally, publish the current paired study and its public inputs, and add the links to the thesis.

## Implemented behavior

The portable pair in `notebooks/04_04_evolution_energetique/` follows the current online study, `main-20261117-rho-div20-online-balance-20261120`. The viewer contains only the retained four-panel balance figure and the global RMS comparison for Simpson every two and four steps. The separate Hamiltonian-variation figure, broken-axis explanation and obsolete two-panel description are removed. H and delta H remain numerical inputs to the residual identities. Historical reconstruction notebooks and scientific archives are preserved.

The calculation uses the reader's personal Modal account and returns a verified ZIP to their computer. Its summaries contain the two residual RMS values and their ratio. Confirmed calls are recovered without another submission. An ambiguous submission stays blocked until the reader supplies an explicit call ID found in their dashboard; that recovery records the ID without spawning a replacement. Worker checkpoints exist during its lifetime and are not promised to survive termination.

Original notebook outputs are preserved under `outputs/colab-backups/04_04_evolution_energetique/` in `calcul-before-colab.ipynb` and `visualisation-before-colab-5c32e4f98eb7.ipynb`. The historical calculation script is byte-identical, with SHA256 `d95df7c9470acfe4f7d2cee53d5c91f1c3079f6f574e64564a51858e172b2afe`.

## Verified results

Seven viewer tests passed. An independent review checked all eight original result hashes, the diagnostic identities and the 48 public diagnostic subsets against their original arrays. The public arrays are lossless float64 values of shape `(5001, 16)`, retaining the original particle IDs and cycle zero. RMS reductions exclude cycle zero: each plotted block contains 100 cycles and eight particles, with centers 50.5 through 4950.5. All 32 plotted series and 16 global comparisons equal the current thesis figure's JSON exactly. The notebook has four code cells and one saved figure with French labels; its format validates successfully.

Eight calculation safety tests and six independent Simpson tests passed. The reduced local check integrated all eight methods and all 84 particles for two cycles at 40 fixed steps per cycle, using the frozen sources and original time plan. Its verified ZIP contains 22 files and 589510 bytes. States, H, delta H, both accumulated integrals and the fine residual match the first two cycles of the original online reference bit for bit for every method. A separate RK2 interruption and checkpoint recovery also reproduces the uninterrupted arrays exactly. These comparisons cover two cycles, not a new validation of the complete 5000-cycle campaign.

The calculation kit retains the original physical HDF5 file, reference, source archive and numerical inputs. Its 21 entries have safe relative names and anonymous archive headers; every recorded file hash was checked independently. The eight input JSON files differ from their originals only in the descriptive `source_path`. Three checked parts concatenate to exactly 600524808 bytes, with SHA256 `c851fd10cd872d90426a15ef157a7216989107de6d450d9699c5605d438d525b`. The calculation manifest SHA256 is `112e180f10e57f320381ea93e98692a3dc40669bf92d8d096f89b601740894c7`. Keeping the full HDF5 file preserves the historical program's fingerprint checks and makes the initial download approximately 600 MB.

## Publication

The three viewer objects and four calculation objects have been uploaded to public B2. The viewer's four code cells were executed locally using their unchanged anonymous HTTPS downloader; manifest, data hashes, identities and exact thesis statistics passed. The viewer manifest SHA256 is `a76603ba09e80eda5b1af794b51af1a8b96aa188f554c51c7bc93aee6cb8115e`.

The [visualization notebook](https://colab.research.google.com/drive/14kK36hblOqiPnGgDMMTO8KId7gXwd7dH) and [calculation notebook](https://colab.research.google.com/drive/1U_zj1rM87b4gYk-hPJslB3CKr2PE2Ejo) are published with reciprocal links. The final Drive bytes were verified and adopted by the publication tool (`adopt_identical` for both, destination and remote listing verified); the existing read-by-link permissions were checked without modification. Both links open the expected notebooks in Colab; no runtime was connected. The visualization is 364260 bytes, SHA256 `453d2f4f2f92a252875717c8abd0154bf4ba7b45a82c4ba8b704f2e425a74990`; the calculation is 33022 bytes, SHA256 `da3ace5203b11ada1eecd13768770f5cba944d8991f82701f77b6f8793dee40b`.

Anonymous HTTPS downloads of the calculation manifest and all three parts were also verified, including the assembled archive and all 20 declared payload files. The public viewer contains three objects totaling 28753949 bytes. Scientific reference archives remain unchanged.

The historical notebook generator delegates to the active builders. Section 4.4 in the French thesis links to both published notebooks; `latexmk -pdf main_fr.tex` completed successfully and generated the 80-page `main_fr.pdf`. The PDF was not opened or rendered. Selection and publication are documented in the personal configuration repository. The durable section-4.4 distinction is recorded in `MEMORY.md` and its Spanish copy, both remaining at 300 lines.

No Google Colab runtime execution or remote Modal calculation was performed for this change. No full campaign was repeated. Quadrature sensitivity remains an empirical diagnostic, not an error bound or proof of trajectory accuracy.

`python3 -B scripts/check_sdd.py` passes with 319 records and 22 requirements. Scoped whitespace checks pass. This check concerns document structure only.