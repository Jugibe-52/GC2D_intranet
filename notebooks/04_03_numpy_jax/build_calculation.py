"""Build the standalone section 4.3 calculation notebook with reviewed helpers."""
from hashlib import sha256
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
KIT_URL = 'https://f003.backblazeb2.com/file/gc2d-colab-public/04_03_numpy_jax/calculation/v1/calculation-kit.tar.xz'
VISUALISATION_URL = 'https://colab.research.google.com/drive/1xVRKtB2AWp_Pqv6COtP1F3z0Ystfa2Bo'


def build(visualisation_url=VISUALISATION_URL):
    manifest = json.loads((ROOT.parents[1] / 'outputs/colab-public/04_03_numpy_jax/calculation/v1/manifest.json').read_text())
    kit_sha256 = manifest['sha256']
    cells = []

    def cell(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        item = dict(cell_type=kind, id=sha256(source.encode()).hexdigest()[:12],
            source=source.splitlines(keepends=True), metadata={})
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', f'''
        # 4.3 — Calculate the NumPy/SciPy and JAX comparison

        Open this notebook in **Google Colab with a standard CPU runtime**. Calculations run on
        **your personal Modal account**; no local GC2D checkout or B2 account is needed.
        Public B2 provides verified, frozen scientific inputs and numerical sources.
        Results are offered as a **ZIP download to your computer** with paired trajectories,
        individual timings, parameters, library versions, checksums and source provenance.

        Create a [Modal account](https://modal.com/) and a personal API token in its settings.
        Modal compute uses your account's allowance and billing; check the
        [current pricing](https://modal.com/pricing) before starting a full run.
        No author's credentials are included. The notebook guides installation and authentication.

        The reference compares **BM4Implicit**, **GaussLegendre4** and **RK2** for **10, 30 and
        90 cycles**, using the 16-particle union of sets 3 and 4, normalized rho=0.015,
        reflected potential, 40 fixed steps and ten saved samples per cycle, and float64.
        One first call and three warmed calls per backend in nine cases give **72 integrations**.
        Both backends run sequentially in the same two-core Modal CPU container for each case
        (4 GiB requested, 8 GiB limit), alternating their order. Cases run sequentially;
        JAX caches are cleared before every case, and repeated trajectories must match bitwise.

        Timings include execution preparation, integration, synchronization and host result
        construction. They exclude input restoration, startup, checks, files and transfers.
        First JAX calls include compilation. Backend distances measure agreement in unwrapped
        coordinates and **do not establish integration accuracy**. Frozen sources and dependency
        versions reproduce the protocol; timings vary with hardware and runtime conditions.

        To inspect the saved reference without calculating, open the
        [visualization notebook]({visualisation_url}). Run the following cells in order.
    ''')
    cell('markdown', '## 1. Install the Modal client\nIntegrations run in the remote container. Colab also uses NumPy to audit the returned arrays.')
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
        try:
            import numpy
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy"], check=True)
        import modal
        print("Modal client ready.")
    ''')
    cell('markdown', '''
        ## 2. Authenticate with your personal Modal account

        Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in Colab's **Secrets** panel (key icon),
        and grant this notebook access. Hidden prompts are available as a fallback.
        Do not paste credentials into code cells. They remain in this runtime's environment;
        the notebook does not write them to saved outputs or result files.
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

        Keep `CYCLES = [10, 30, 90]` and `REPETITIONS = 3` for the reference protocol.
        Set `CYCLES = [1]` for a short workflow check across all three methods; its timings
        are not thesis results. A custom selection can contain up to three distinct durations
        from 1 to 90 cycles. Set `LAUNCH_CALCULATION = True` when ready.

        Keep the generated `RUN_ID` and download the receipt. Choose a new identifier before
        starting a different experiment. After a disconnected session, enter your old
        `RUN_ID`, the same settings and `RESUME_CALL_ID` to retrieve the existing call.
        Modal call results have limited retention: download promptly. Colab runtime storage
        is temporary, and saving the notebook does not save its result ZIP.
    ''')
    cell('code', '''
        from datetime import datetime, timezone
        from pathlib import Path
        from uuid import uuid4

        CYCLES = [10, 30, 90]
        REPETITIONS = 3  # @param {type:"integer"}
        LAUNCH_CALCULATION = False  # @param {type:"boolean"}
        RUN_ID = globals().get("RUN_ID", "reader-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
        RESUME_CALL_ID = ""  # @param {type:"string"}
        WORKSPACE = Path.cwd() / "gc2d-colab-43"
        WORKSPACE.mkdir(exist_ok=True)
        print(f"Run: {RUN_ID}; durations: {CYCLES}; {6 * len(CYCLES) * (REPETITIONS + 1)} integrations.")
    ''')
    cell('markdown', f'''
        ## 4. Load and verify the reproduction kit

        Download approximately {manifest['bytes'] / 1e6:.1f} MB from public B2. Verify the pinned SHA-256
        before loading archived Python inputs. Reviewed helper code is embedded below so
        this notebook needs no checkout. Only the historical source-path provenance is
        normalized for publication; numerical arrays, methods and execution settings are preserved.
        The kit records the original and public input fingerprints and strips personal tar headers.
    ''')
    runtime = (ROOT / 'calculation_runtime.py').read_text()
    worker = (ROOT / 'calculation_worker.py').read_text()
    cell('code', f'''
import importlib
import json
import sys

KIT_URL = {KIT_URL!r}
KIT_SHA256 = {kit_sha256!r}
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
worker = importlib.reload(worker)
print("Inputs and frozen sources verified. No calculation has started.")
''')
    cell('markdown', '''
        ## 5. Launch once, or recover the recorded call

        Keep the receipt. A confirmed call is reused if this cell runs again.
        If submission stops before recording a call ID, inspect your
        [Modal dashboard](https://modal.com/apps) before starting another experiment.
        This guard prevents automatic duplicate integrations after an ambiguous submission.
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
        ## 6. Retrieve and verify the completed result

        This cell waits for the saved call and never launches a new one. A timeout can be
        retried with the same receipt. If Modal reports an expired result, check any ZIP
        already downloaded. The verified archive includes both trajectories per case,
        paired timings, metadata, the frozen source and a complete checksum manifest.
    ''')
    cell('code', '''
        from statistics import median

        RESULT_ZIP = RUN_DIRECTORY / (RUN_ID + ".zip")
        if RESULT_ZIP.exists():
            result_bytes = RESULT_ZIP.read_bytes()
        else:
            result_bytes = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=3600)
        study, timings = runtime.verify_results(result_bytes, settings, inputs)
        temporary_zip = RESULT_ZIP.with_suffix(".tmp")
        temporary_zip.write_bytes(result_bytes)
        temporary_zip.replace(RESULT_ZIP)
        print(f"Verified {len(timings['cases'])} cases; result archive: {len(result_bytes):,} bytes.")
        print(f"{'Case':24} {'Library':12} {'First (s)':>12} {'Median (s)':>12} {'Max distance':>14}")
        for case, pair in timings["cases"].items():
            for backend in runtime.BACKENDS:
                rows = [r for r in pair["measurements"] if r["backend"] == backend]
                first = next(r["seconds"] for r in rows if r["repetition"] == 0)
                warm = [r["seconds"] for r in rows if r["repetition"] > 0]
                distance = pair["agreement"]["maximum_distance"]
                print(f"{case:24} {backend:12} {first:12.6f} {median(warm):12.6f} {distance:14.6e}")
    ''')
    cell('markdown', '''
        ## 7. Download to your computer

        Allow the browser download and save the ZIP and receipt before closing the runtime.
        No personal bucket or Modal Volume is needed. This reader-download destination is
        deliberate: a reader has no write access to the author's scientific B2 archive.
        The ZIP retains numerical arrays and metadata for your own analysis. The separate
        visualization notebook displays the saved thesis reference without rerunning it.
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
        'gc2d': dict(kit_url=KIT_URL, kit_sha256=kit_sha256, reference_run_id=manifest['reference_run_id'])})
    destination = ROOT / 'calcul.ipynb'
    backup = ROOT.parents[1] / 'outputs/colab-backups/04_03_numpy_jax/calcul-before-colab.ipynb'
    if destination.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(destination.read_bytes())
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visualisation-url', default=VISUALISATION_URL)
    args = parser.parse_args()
    print(build(args.visualisation_url))
