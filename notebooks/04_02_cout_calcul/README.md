# 04_02 — Method runtime

The standalone notebook loads its verified public dataset from [B2](https://f003.backblazeb2.com/file/gc2d-colab-public/04_02_cout_calcul/jax-20261118-rho0015-c500/v1/manifest.json).
Its saved outputs were produced by a local Python execution using those HTTPS files without authentication.
This validates public B2 access and notebook execution locally; it does not claim execution inside a Google Colab runtime.

- [calcul.ipynb](calcul.ipynb) executes or recovers the eight-method JAX study and preserves individual timings, results, parameters and provenance.
- [visualisation.ipynb](visualisation.ipynb) downloads a small verified public B2 dataset and displays the runtime comparison figure in Colab or Jupyter, without repeating integrations.

This folder holds the canonical notebooks themselves, including their saved outputs.
Notebook prose is English, and thesis figure text is French.
The calculation notebook is standalone in Colab or Jupyter: only the Modal client is installed locally.
Its first cells guide personal Modal authentication, duration and repetition settings, explicit launch and ZIP download to the reader's computer.
A pinned reproduction kit supplies the original archived source and eight input payloads; numerical dependencies are installed only in a Python 3.12 Modal worker.
The reader needs no GC2D checkout, B2 account or Modal Volume.
This reader-download destination is the explicit user-authorized exception to the default B2 result-storage policy.
The visualization notebook is standalone: it uses only Python's standard library, NumPy and Matplotlib, with no secrets, JAX, rclone or repository installation.
Run all cells to download the published data and display the figure; its saved outputs also allow reading without executing it.
The optional export cell saves PDF, PNG and JSON provenance in a directory selected within the current runtime; in Colab, download them from the Files panel before the runtime expires.

Both notebooks refer to the original study `studies/method_timings/jax-20261118-rho0015-c500`; new reader calculations receive their own run identifier.
The scientific destination remains `gc2d_data:gc2d-notebooks-data/studies/method_timings/jax-20261118-rho0015-c500`; `GC2D_codigo/outputs/` holds local recovery copies.
The visualization data publication contains only the unchanged `study.json`, unchanged `timings.json`, and a dedicated `manifest.json` that lists their byte counts and SHA-256 digests.
The notebook pins the manifest digest, checks both data files and all 32 timing records, and reproduces the same medians, minimum–maximum ranges and fixed RK2 denominator as the original figure generator.
It reports the original determinism audit as recorded provenance; it does not download or re-audit the trajectory archives.
The measurements describe the original JAX CPU worker, not the Colab runtime.

[visualisation.py](visualisation.py) is the reviewable percent-format source, including the public URL and pinned manifest digest.
Regenerate the notebook after source changes with `python notebooks/04_02_cout_calcul/build_visualisation.py` from the code repository; saved outputs survive only when the complete source, including its data URL and manifest digest, is unchanged.
Run the focused offline archive-parity and integrity checks with `.venv/bin/python notebooks/04_02_cout_calcul/test_visualisation.py`.
Those checks require the original local timing archive and the existing thesis figure's JSON provenance but perform no simulation or network transfer.

Prepare or verify an identical publication bundle with `python notebooks/04_02_cout_calcul/prepare_public_data.py /path/to/new/version/directory`.
The script checks the original archive manifest and publication receipt, preserves both JSON files byte for byte, creates a deterministic public manifest, and refuses to overwrite a different version.
Its printed manifest digest must match `EXPECTED_MANIFEST_SHA256`; the script does not create a bucket or upload files.
Execute all notebook cells offline against that exact bundle with `MPLBACKEND=Agg python notebooks/04_02_cout_calcul/validate_visualisation.py /path/to/new/version/directory --save-outputs`.
This substitutes local file reads for HTTP transport while retaining the pinned manifest, integrity and protocol checks; it labels the saved outputs as offline and does not prove public B2 or Colab access.
After configuring the real public URL and rebuilding the notebook, execute all cells with `MPLBACKEND=Agg python notebooks/04_02_cout_calcul/validate_visualisation.py --live --save-outputs`.
This mode uses the notebook's unchanged HTTPS downloader, verifies the actual public objects and records the execution time, public URL and manifest digest with its saved outputs.
It proves local Python execution using public B2 data; it does not claim execution inside a Google Colab runtime.
Choose either an offline bundle directory or `--live`; a failed execution leaves the notebook and its previous saved outputs unchanged.

## Portable calculation

Regenerate `calcul.ipynb` with `.venv/bin/python notebooks/04_02_cout_calcul/build_calculation.py`. The builder embeds the reviewed [runtime](calculation_runtime.py) and [worker](calculation_worker.py); it clears outputs so credentials and test results are never published. The default remains 500 cycles and three cached repetitions for each of eight methods. A shorter duration is labelled by its saved configuration and is not the thesis experiment.

[prepare_calculation_data.py](prepare_calculation_data.py) verifies the original archive receipt and file hashes, including all payload fingerprints, and builds an immutable 3,790,204-byte reproduction kit. Its SHA-256 is `c7eac5a241a834bf910ca7cd910a46918e4c7168eebd678a496a19121d888375`. The kit contains no credentials and preserves the original numerical source and inputs. The source archive is deliberately frozen rather than replaced by later library revisions.

[validate_calculation.py](validate_calculation.py) runs a disposable one-cycle copy with all eight methods and 32 integrations on the tester's existing Modal account. It skips the interactive credential cell, leaves the canonical 500-cycle notebook unchanged and checks confirmed-call reuse. Use `--kit /path/to/calculation-kit.tar.xz` for local input transport or omit it to check the public HTTPS endpoint. This launches real compute only if the workspace has no saved receipt. Six offline integrity and recovery checks run with `.venv/bin/python notebooks/04_02_cout_calcul/test_calculation.py`.

A confirmed receipt is reused; an interrupted, unconfirmed submission requires checking the Modal dashboard. The resulting ZIP includes complete arrays, timings, parameters, archived sources and a manifest. Download it and the receipt before closing Colab; neither the `.ipynb` file nor the temporary runtime is permanent result storage. Publication and validation status is recorded in [the change record](../../specs/changes/2026-11-27-colab-calculation-42/results.md).
