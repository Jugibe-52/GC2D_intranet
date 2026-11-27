# Execution record

The user accepted and authorized creation/publication of the section-4.5 Colab pair and its thesis links on 2026-11-27. The pair is published and verified, its links are in the French thesis, and the PDF compiles successfully.

The browser attachment failed before opening a page because the desktop integration requires a network policy. The existing B2 connection can list only the private scientific bucket. No credentials or permissions were changed. The viewer therefore carries a verified small reference snapshot; calculation reuses existing public B2 inputs. Drive publication used the existing rclone connection, with bounded recoverable retries for intermittent shared-client quota errors. The final remote bytes and IDs were verified.

## Viewer validation

The preparation audited 21 complete archive pairs, including 6108791916 solution bytes and 147 diagnostic arrays. Source sizes/SHA256, shapes, finite values, diagnostic identities and five reductions agree with the historical summary. All eight plotted series equal the retained thesis figure records. Four notebook code cells execute locally from the embedded snapshot and save two figures with French text. Eight focused tests pass, including corruption rejection, radius semantics, the unavailable case, plotted series and label layout. The prior local viewer is preserved in outputs/colab-backups/04_05_correction_projection.

The embedded bundle contains manifest.json (404 bytes; SHA256 c88767c8bf12af1c4fd17775598f21857d4bcd5206cae568075c804aa2f83c8a), summary.json (13575 bytes) and provenance.json (54138 bytes). It contains statistical reductions and their verification record, not the original diagnostic arrays. Notebook execution verifies this pinned snapshot; it does not independently repeat the original archive audit. No new public B2 objects were uploaded.

## Calculation validation

The portable calculation uses the already-public section-4.4 kit, pinned by its manifest and archive hashes. An independent audit recalculated the frozen numerical-source fingerprint and checked exact equality of the 84 initial states and complete time plans across all 21 available projection cases. The downloaded result retains particle IDs, colors, physical parameters, signed sampled multipliers, all-step joint Newton diagnostics, timings and source provenance. The original kernels and archived inputs remain unchanged.

Local validation integrated both methods for one cycle at normalized radii 0 and 0.015, with all 84 particles, 40 fixed steps and ten saved multiplier samples. All four reduced cases completed, and their individual and combined ZIPs passed verification. In particular, reduced ABBA2 at radius zero succeeds; its unavailable status in the 5000-cycle historical reference remains unchanged.

For the three cases with complete historical archives, an independent read-only comparison found byte-for-byte equality over the first cycle for all ten checked arrays: states `(11, 168)`, saved times `(11,)`, signed multipliers `(10, 2, 84)`, and seven histories of 40 entries (Newton iterations, residuals, tolerances, multiplier infinity norms, step starts, sizes and ends). Every maximum absolute difference is zero. These first-cycle comparisons do not establish agreement over 5000 cycles. Short local timings include compilation and are not comparable with full-campaign timings.

Ten focused calculation tests pass. They cover sampled Euclidean reductions, every-step convergence, explicit launch, reuse of confirmed calls, refusal to resubmit ambiguous calls, recovery, corrupt ZIPs and explicit failure records. The result reader also validates source and unit metadata, exact full-run clocks, block timing counts and totals, and floating reductions within a relative tolerance of `2e-13`. This tolerance only accommodates reduction roundoff across NumPy environments; it is not a trajectory-error estimate. Calls use the reader's Modal account, with no automatic retry, and provide downloadable receipts and results.

These checks ran locally; no execution inside a Google Colab runtime or on Modal, and no new full campaign, is claimed. The new driver never replaces historical reference availability or statistics. The viewer's saved outputs come from its actual local snapshot execution.

## Publication and thesis integration

The personal Colab publication tool created the calculation and updated the viewer in `COLAB NOTEBOOKS/MEMOIRE_TFM`, retaining the viewer's ID and limiting publication to the 4.5 pair. The destination listing and both reciprocal links were checked. Anonymous Drive HTTP GET requests returned status 200 and the exact final local bytes for both notebooks, confirming public read access without permission changes.

| Notebook | Drive ID | Bytes | SHA-256 | MD5 |
| --- | --- | ---: | --- | --- |
| [Calculation](https://colab.research.google.com/drive/1dlmACJ_X4mgxi73jNzgoiD6VG926mh_Q) | `1dlmACJ_X4mgxi73jNzgoiD6VG926mh_Q` | 100930 | `998075ba9bf1e2698a989bfbc0f80544cae4d9477872afe976e4e4a22af91beb` | `500b9b3da8a94dd4617bbd736078553a` |
| [Visualization](https://colab.research.google.com/drive/1CIgaOG-slC8KnXpg0gGbzdvbwX_keNyh) | `1CIgaOG-slC8KnXpg0gGbzdvbwX_keNyh` | 261435 | `2bfbf2f0d1ecf5280f357bf5092d720c493fbe035a65e97d6b4cb9111ef5d58d` | `3123e631d52ccfd239f2c59f513ec4b5` |

The historical thesis generator's `--notebook` route was checked after its delegation fix and preserves the portable viewer's hash. Section 4.5 contains both links, and the chapter introduction explains the embedded reference snapshot. From the thesis root, `latexmk -pdf main_fr.tex` exited with status 0 and generated `main_fr.pdf` with 81 pages and 12116194 bytes. Its log contains no warning, undefined reference or error marker. The PDF was neither opened nor rendered for this check.

The document check `.venv/bin/python -B scripts/check_sdd.py`, run from the code root, passed: 324 records and 22 requirements. This read-only checker excludes Spanish reading copies and validates document structure; it neither runs scientific checks nor proves publication.
