"""Build the standalone calculation notebook, embedding its reviewed helpers."""
from hashlib import sha256
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
KIT_URL = 'https://f003.backblazeb2.com/file/gc2d-colab-public/04_02_cout_calcul/calculation/v1/calculation-kit.tar.xz'
KIT_SHA256 = 'c7eac5a241a834bf910ca7cd910a46918e4c7168eebd678a496a19121d888375'


def build():
    cells = []

    def cell(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        item = dict(cell_type=kind, id=sha256(source.encode()).hexdigest()[:12],
            source=source.splitlines(keepends=True), metadata={})
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''
        # 4.2 — Calculate the eight-method runtime comparison

        Run this notebook in **Google Colab with a standard CPU runtime**. Calculations run on
        **your own Modal account**, not on Colab. No local GC2D installation or B2 account is needed.
        The author's public B2 bucket supplies verified, frozen inputs and executable sources.
        New results are offered as a **ZIP download to your computer**, including all eight
        trajectories, individual timings, parameters, checksums and source provenance.

        Create a [Modal account](https://modal.com/) and a personal API token in your Modal settings.
        The [Starter offer](https://modal.com/pricing) currently includes **30 USD of compute credit
        per month** (checked 27 November 2026); resource use beyond the included allowance can be billed.
        Installation, authentication and calculation settings are configured below.

        The reference experiment uses 40 particles, reflected potential, normalized rho=0.015,
        40 fixed steps and ten saved samples per cycle, float64, 500 cycles and eight methods.
        One first call plus three cached repetitions per method means **32 integrations**.
        All methods run sequentially in one two-core Modal CPU container (4 GiB requested, 8 GiB limit).
        Timing includes preparation, integration, synchronization and host results; it excludes
        container startup, input decoding, audits, transfers and file saving. First calls include
        compilation. The original sources and dependencies are frozen to reproduce the protocol;
        execution times can differ with the available hardware.

        To view the reference figure without calculating, open the
        [visualization notebook](https://colab.research.google.com/drive/1_AFGcTjOxlplLJdRxEfFquAkIQ_LiToS).
        Run the cells in order. Calculation is enabled explicitly in the settings cell.
    ''')
    cell('markdown', '## 1. Install the Modal client\nThe numerical libraries are installed in the remote container, not in Colab.')
    cell('code', '''
        import importlib.metadata
        import subprocess
        import sys

        try:
            installed = importlib.metadata.version("modal")
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != "1.6.0":
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "modal==1.6.0"], check=True)
        import modal
        print("Modal client ready.")
    ''')
    cell('markdown', '''
        ## 2. Authenticate with your personal Modal account

        In Colab's **Secrets** panel (key icon), add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
        and grant this notebook access. Alternatively, enter them in the hidden prompts below.
        Do not paste credentials into code cells. They are held in the runtime environment;
        they are never added to the downloaded results or saved notebook outputs.
        Existing environment credentials are reused when running locally.
    ''')
    cell('code', '''
        import getpass
        import os

        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
            if not os.environ.get(key):
                value = None
                try:
                    from google.colab import userdata
                    value = userdata.get(key)
                except Exception:
                    pass
                if not value:
                    value = getpass.getpass(f"{key}: ")
                if not value:
                    raise ValueError("Both personal Modal credentials are required.")
                os.environ[key] = value.strip()
                del value
        print("Credentials loaded for this runtime; no calculation has started.")
    ''')
    cell('markdown', '''
        ## 3. Choose the calculation

        Keep **500 cycles and 3 repetitions** to reproduce the reference protocol. A shorter
        duration such as 1 cycle checks the workflow but does not reproduce the thesis timings.
        Keep the generated run identifier to recover the same calculation. Choose a new identifier
        before starting another experiment. Set `LAUNCH_CALCULATION = True` when ready.

        After a disconnected Colab session, enter the previous `RUN_ID`, settings and
        `RESUME_CALL_ID` from the receipt to recover that call without repeating the calculation.
        Modal call results have limited retention; download the ZIP promptly. The `.ipynb` file
        itself does not contain the result archive, and Colab runtime files are temporary.
    ''')
    cell('code', '''
        from datetime import datetime, timezone
        from pathlib import Path
        from uuid import uuid4

        CYCLES = 500  # @param {type:"integer"}
        REPETITIONS = 3  # @param {type:"integer"}
        LAUNCH_CALCULATION = False  # @param {type:"boolean"}
        RUN_ID = globals().get("RUN_ID", "reader-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
        RESUME_CALL_ID = ""  # @param {type:"string"}
        WORKSPACE = Path.cwd() / "gc2d-colab-42"
        WORKSPACE.mkdir(exist_ok=True)
        print(f"Run: {RUN_ID}; {CYCLES} cycles; {8 * (REPETITIONS + 1)} integrations.")
    ''')
    cell('markdown', '''
        ## 4. Load the verified reproduction kit

        This downloads about 3.8 MB from public B2. Checksums are verified before the archived
        Python inputs are used. The helper code below is embedded so no repository checkout is
        needed; its reviewable sources accompany the original notebook.
    ''')
    runtime = (ROOT / 'calculation_runtime.py').read_text()
    worker = (ROOT / 'calculation_worker.py').read_text()
    cell('code', f'''
import importlib
import json
import sys

KIT_URL = {KIT_URL!r}
KIT_SHA256 = {KIT_SHA256!r}
(WORKSPACE / "calculation_runtime.py").write_text({runtime!r})
(WORKSPACE / "calculation_worker.py").write_text({worker!r})
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
importlib.invalidate_caches()
import calculation_runtime as runtime
runtime = importlib.reload(runtime)
settings = runtime.check_settings(CYCLES, REPETITIONS, RUN_ID)
inputs = runtime.download_kit(KIT_URL, KIT_SHA256, WORKSPACE / "kit")
import calculation_worker as worker
print("Original inputs and sources verified. No calculation has started.")
''')
    cell('markdown', '''
        ## 5. Launch once, or recover the recorded call

        Keep a copy of the receipt shown below. If submission is interrupted before a call ID
        is recorded, inspect your [Modal dashboard](https://modal.com/apps) before launching again.
        A confirmed call is reused even if this cell is executed again.
    ''')
    cell('code', '''
        from IPython.display import display, FileLink

        if not LAUNCH_CALCULATION and not RESUME_CALL_ID:
            raise RuntimeError("Set LAUNCH_CALCULATION=True in cell 3, then run cells 3 to 7.")
        RUN_DIRECTORY = WORKSPACE / "results" / RUN_ID
        RECEIPT = RUN_DIRECTORY / "receipt.json"
        receipt = runtime.launch_or_recover(worker, RECEIPT, settings, KIT_SHA256, RESUME_CALL_ID)
        print(json.dumps(receipt, indent=2))
        display(FileLink(str(RECEIPT)))
        try:
            from google.colab import files
            files.download(str(RECEIPT))
        except ImportError:
            pass
    ''')
    cell('markdown', '''
        ## 6. Retrieve and verify the completed results

        Wait for the recorded call. Rerunning this cell retrieves the same result; it never
        launches a calculation. A timeout can be retried with the same receipt. If Modal reports
        an expired result, it cannot be recovered from the notebook: check any ZIP already downloaded.
        The archive contains `study.json`, `timings.json`, `runs/<method>/` with numerical arrays
        and metadata, frozen sources, input fingerprints and a checksum manifest.
    ''')
    cell('code', '''
        from statistics import median

        RESULT_ZIP = RUN_DIRECTORY / (RUN_ID + ".zip")
        if RESULT_ZIP.exists():
            result_bytes = RESULT_ZIP.read_bytes()
        else:
            result_bytes = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=3600)
        study, timings = runtime.verify_results(result_bytes, settings)
        if study["original_input_files"] != inputs["files"]:
            raise ValueError("The recovered result used different original inputs.")
        temporary_zip = RESULT_ZIP.with_suffix(".tmp")
        temporary_zip.write_bytes(result_bytes)
        temporary_zip.replace(RESULT_ZIP)
        print(f"Verified {len(timings['measurements'])} integrations; archive: {len(result_bytes):,} bytes.")
        print(f"{'Method':20} {'First (s)':>12} {'Median (s)':>12} {'Minimum (s)':>12} {'Maximum (s)':>12}")
        for name in runtime.METHODS:
            rows = [r for r in timings["measurements"] if r["method"] == name]
            first = next(r["seconds"] for r in rows if r["repetition"] == 0)
            warm = [r["seconds"] for r in rows if r["repetition"] > 0]
            print(f"{name:20} {first:12.6f} {median(warm):12.6f} {min(warm):12.6f} {max(warm):12.6f}")
    ''')
    cell('markdown', '''
        ## 7. Download to your computer

        Run this cell and allow the browser download. Save the ZIP and receipt on your computer
        before closing the Colab runtime. Results are not uploaded to the author's B2 bucket.
        No personal bucket or Modal Volume is required.
    ''')
    cell('code', '''
        try:
            from google.colab import files
            files.download(str(RESULT_ZIP))
        except ImportError:
            display(FileLink(str(RESULT_ZIP)))
        print("Download offered. Check that the ZIP is saved on your computer.")
    ''')
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': dict(display_name='Python 3', language='python', name='python3'),
        'language_info': dict(name='python'), 'colab': dict(provenance=[]),
        'gc2d': dict(kit_url=KIT_URL, kit_sha256=KIT_SHA256, reference_run_id='jax-20261118-rho0015-c500')})
    destination = ROOT / 'calcul.ipynb'
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    return destination


if __name__ == '__main__':
    print(build())
