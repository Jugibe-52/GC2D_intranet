# Completed third-set calculation

On 2026-11-16 the user authorized creation and calculation of the new third-set notebook with `&`, following the request for an image like the supplied BM4Implicit example. The existing campaign defaults were stated before implementation. The figure was included in the thesis using the announced left-figure/right-text layout after the optional preference question.

Created and executed `notebooks/third_set_poincare.ipynb`. It reads the current eight-particle dataset, calculates BM4Implicit returns and exports French-labelled PNG/PDF figures. Calculation parameters remain editable; complete matching archives can be reused. Documentation is in `docs/studies/third-set-poincare.md` with a Spanish reading copy.

A disposable three-cycle run completed 120 fixed steps and verified the calculation, archive reuse and plot export. The requested full run completed 200000 fixed steps over 5000 cycles in 1378.57 seconds, including JAX compilation. Its archive is `data/trajectory/third_set_poincare/bm4implicit_2e069839791cac95.npz`, with a JSON metadata companion. It contains finite float64 states of shape `(5001, 16)`, unchanged initial positions and IDs, exact whole-cycle output times and zero output interpolation. The executed notebook passes notebook-schema checks and contains no error outputs.

The plot represents 40000 returns from the eight particles. Of these, 1897 lie outside the displayed unit square; all remain in the archive. Saved coordinates range from approximately -1.986 to 2.982 horizontally and -0.990 to 1.987 vertically. No wrapping or modulo-one folding was applied. The archive hash and its particle-data fingerprint were checked against the figure provenance. The exported PNG was inspected for the requested square scatter style and readable labels.

The thesis uses `figuras/tercer_conjunto_poincare_fr.pdf`, with PNG and JSON companions. The caption explicitly reports the excluded viewport points. Chapter 4 links to the new notebook. `latexmk -pdf main.tex` regenerated the 67-page `main.pdf`; the final log has no errors, unresolved references or line-layout warnings. The thesis PDF was not opened or rendered for this check.

The numerical image does not establish chaos or long-time convergence; the dataset remains a candidate selection. Existing particle records and numerical implementation files were unchanged. No dependencies were installed and no remote calculation, publication, commit or synchronization was performed.
