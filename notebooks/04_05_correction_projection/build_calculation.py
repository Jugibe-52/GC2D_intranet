"""Build the standalone section-4.5 personal-Modal calculation notebook."""
from hashlib import sha256
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT.parents[1] / 'outputs/colab-public/04_05_correction_projection/calculation/v1'
BASE = ROOT.parents[1] / 'outputs/colab-public/04_04_evolution_energetique/calculation/v1'
KIT_URL = 'https://f003.backblazeb2.com/file/gc2d-colab-public/04_04_evolution_energetique/calculation/v1/manifest.json'
VISUALISATION_URL = 'https://colab.research.google.com/drive/1CIgaOG-slC8KnXpg0gGbzdvbwX_keNyh'


def build(visualisation_url=VISUALISATION_URL):
    protocol_bytes = (PUBLIC / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest_bytes = (BASE / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    previous = json.loads((ROOT.parent / '04_03_numpy_jax/calcul.ipynb').read_text())
    common = [c for c in previous['cells'] if c['cell_type'] == 'code']
    cells = []

    def cell(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        value = dict(cell_type=kind, id=sha256(source.encode()).hexdigest()[:12],
            metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell('markdown', f'''
        # 4.5 — Calculate projection corrections

        Open this notebook in **Google Colab with a standard CPU runtime**. It integrates
        **new trajectories on your personal Modal account**, with no local GC2D checkout
        and no B2 credentials. The historical numerical kernels, measured potential and
        initial data come from the verified public frozen kit already used by section 4.4.
        Your results are saved as checked **ZIP downloads to your computer**.

        The reference campaign is `main-20261117-rho-div20`. The full calculation uses
        **ABBA2Implicit and BM4Implicit**, eleven normalized radii rho=i/400 (i=0,...,10),
        **84 particles**, **5000 cycles**, **40 fixed steps per cycle**, and **ten samples per cycle**.
        Initial positions, IDs and colors are preserved. Coordinates are unwrapped in the
        reflected potential with spatial period two; rho is already in base-cell widths.

        The signed two-component projection multiplier is saved after steps 4,8,... .
        We calculate the median and maximum of its **Euclidean norm** over those sampled
        steps and all particles. Newton mean/max correction counts use **every step**;
        each count is one solve for the joint batch of 84 particles. The sampled maximum
        is not the maximum over all steps and the multiplier is not a trajectory-error estimate.

        The [visualization notebook]({visualisation_url}) displays the archived reference
        without re-integration: 21 available cases and the unavailable rho=0 ABBA2Implicit case.
        This notebook attempts every selected case and explicitly records any observed
        nonlinear failure. A new run never replaces that archival availability record.
        Agreement of new full trajectories with the historical campaign is not established.

        Create a [Modal account](https://modal.com/) and personal API token. Remote work
        uses your allowance and billing; check [current pricing](https://modal.com/pricing).
        Each case requests two CPU cores and 4 GiB memory (12 GiB limit), with a four-hour
        timeout and no retry. At most two cases run concurrently. The full 22-case study
        can take several hours and produces multiple gigabytes before compression.
        Keep Colab disk space available and download the ZIPs before closing the runtime.
    ''')
    cell('markdown', '## 1. Install the client\nNumerical dependencies are installed in each remote worker; NumPy verifies the returned histories locally.')
    cell('code', ''.join(common[0]['source']))
    cell('markdown', '''
        ## 2. Authenticate with your personal Modal account

        Store `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in Colab's **Secrets** panel and
        allow notebook access. Hidden prompts provide a fallback. Tokens stay in the
        runtime environment and are not stored in results or notebook outputs.
    ''')
    cell('code', ''.join(common[1]['source']))
    cell('markdown', '''
        ## 3. Set the selection and explicitly enable launch

        The reference uses `CYCLES=5000`, `BLOCK_CYCLES=100` and `RHO_INDICES=list(range(11))`.
        For a small workflow check use `CYCLES=1`, `BLOCK_CYCLES=1`, `RHO_INDICES=[0,6]`.
        Both methods still integrate all 84 particles at the same fixed step. Such a
        check is a new reduced run, not the thesis reference.

        Set `LAUNCH_CALCULATION=True` only when ready. Keep `RUN_ID` and the downloaded
        receipt file. After disconnection, restore the same identifier/settings and set
        `RESUME_CALLS` to the recorded mapping of case keys to `fc-...` identifiers.
        Each case has its own call and result, so recovery never needs a replacement run.
        Block processing limits memory; it is not a persistent remote checkpoint.
    ''')
    cell('code', '''
        from datetime import datetime, timezone
        from pathlib import Path
        from uuid import uuid4

        CYCLES = 5000  # @param {type:"integer"}
        BLOCK_CYCLES = 100  # @param {type:"integer"}
        RHO_INDICES = list(range(11))
        LAUNCH_CALCULATION = False  # @param {type:"boolean"}
        RUN_ID = globals().get("RUN_ID", "reader-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
        RESUME_CALLS = {}  # {"rho-0.015-BM4Implicit": "fc-...", ...}
        WORKSPACE = Path.cwd() / "gc2d-colab-45"
        WORKSPACE.mkdir(exist_ok=True)
        print(f"Run {RUN_ID}: {2 * len(RHO_INDICES)} cases; {CYCLES} cycles per case.")
    ''')
    cell('markdown', '''
        ## 4. Download and verify the frozen inputs

        The shared download is approximately **601 MB**, split into three immutable
        public objects. The manifest, each part, the complete archive and every extracted
        input are checked. This is the exact physical potential and source archive from
        the selected main campaign. We use only the original initial state and time plans
        from the section-4.4 bundle; its energy-balance program is never executed here.

        All 21 available projection cases have bitwise-identical initial states and
        archived time plans. That equality was verified against their original checksums
        when building the embedded projection protocol. The unavailable case uses the
        same planned clocks for its new attempt. Particle IDs, colors, source hashes
        and the reference failure record remain available in the verified inputs.
    ''')
    cell('code', f'''
import importlib
import json
import sys

KIT_URL = {KIT_URL!r}
MANIFEST_SHA256 = {sha256(manifest_bytes).hexdigest()!r}
KIT_SHA256 = {manifest['sha256']!r}
PROTOCOL_SHA256 = {sha256(protocol_bytes).hexdigest()!r}
(WORKSPACE / "calculation_runtime.py").write_text({(ROOT / 'calculation_runtime.py').read_text()!r})
(WORKSPACE / "calculation_worker.py").write_text({(ROOT / 'calculation_worker.py').read_text()!r})
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
importlib.invalidate_caches()
import calculation_runtime as runtime
runtime = importlib.reload(runtime)
settings = runtime.check_settings(CYCLES, BLOCK_CYCLES, RUN_ID, RHO_INDICES)
runtime.download_kit(KIT_URL, MANIFEST_SHA256, KIT_SHA256, WORKSPACE / "kit")
(WORKSPACE / "kit/projection-protocol.json").write_text({protocol_bytes.decode()!r})
protocol, inputs = runtime.verify_protocol(WORKSPACE / "kit", PROTOCOL_SHA256)
import calculation_worker as worker
worker = importlib.reload(worker)
print("Frozen sources, physical potential, initial state and time plans verified; no calculation has started.")
''')
    cell('markdown', '''
        ## 5. Launch each case once or recover its recorded call

        All call IDs are recorded immediately. Rerunning this cell reuses confirmed calls.
        An interrupted submission is marked ambiguous and never silently resubmitted:
        inspect the [Modal dashboard](https://modal.com/apps), then add the confirmed ID
        to `RESUME_CALLS`. See Modal's [job-queue recovery guide](https://modal.com/docs/guide/job-queue).
        Download the receipt file; notebook saving alone does not preserve runtime files.
    ''')
    cell('code', '''
        from IPython.display import display, FileLink

        RUN_DIRECTORY = WORKSPACE / "results" / RUN_ID
        receipts = runtime.launch_or_recover(worker, RUN_DIRECTORY, settings, protocol,
            PROTOCOL_SHA256, allow_launch=LAUNCH_CALCULATION, resume_calls=RESUME_CALLS)
        RECOVERY = RUN_DIRECTORY / "receipts.json"
        print(json.dumps({key: record["call_id"] for key, record in receipts.items()}, indent=2))
        try:
            from google.colab import files
            files.download(str(RECOVERY))
        except ImportError:
            display(FileLink(str(RECOVERY)))
    ''')
    cell('markdown', '''
        ## 6. Retrieve each result and verify the numerical diagnostics

        This cell waits on existing calls. It never launches a replacement. Completed
        ZIPs are saved immediately and reused on rerun. A wait timeout leaves the recorded
        calls recoverable; rerun this cell later. Expired results need an already downloaded
        ZIP or an explicit new run. Cloud result retention and Colab storage are temporary.

        Each complete ZIP contains the sampled physical trajectories, signed multipliers,
        all-step Newton counts/residuals/tolerances, exact clock arrays, parameters, timings,
        numerical-driver source and checksums. Failed cases contain a failure receipt,
        including the first failed step, and are excluded from completed-case statistics.
        Numerical timings include compilation and diagnostics and exclude network transfers.
    ''')
    cell('code', '''
        for case in runtime.selected_cases(protocol, settings):
            key = case["case_key"]
            result_path = RUN_DIRECTORY / (key + ".zip")
            if result_path.exists():
                result_bytes = result_path.read_bytes()
            else:
                try:
                    result_bytes = modal.FunctionCall.from_id(receipts[key]["call_id"]).get(timeout=3600)
                except modal.exception.OutputExpiredError as error:
                    raise RuntimeError(f"Result expired for {key}. Restore a downloaded ZIP; no replacement was launched.") from error
                except modal.exception.TimeoutError as error:
                    raise RuntimeError(f"Still waiting for {key}. Rerun this cell to recover the same call.") from error
            receipt = runtime.verify_case_result(result_bytes, case, settings, protocol, PROTOCOL_SHA256)
            temporary = result_path.with_suffix(".tmp")
            temporary.write_bytes(result_bytes)
            temporary.replace(result_path)
            print(key, receipt["status"], receipt.get("summary", receipt.get("reason")))
        RESULT_ZIP, summary = runtime.combine_results(RUN_DIRECTORY, settings, protocol, PROTOCOL_SHA256)
        print("Verified", sum(c["status"] == "complete" for c in summary["cases"]),
            "complete cases out of", len(summary["cases"]), "; combined ZIP:", RESULT_ZIP.name)
    ''')
    cell('markdown', f'''
        ## 7. Download and preserve the results

        Save the combined ZIP and receipt before closing Colab. The ZIP contains the small
        `summary.json`, study settings, checksums and individually verified case ZIPs.
        Individual files are also available in Colab's file browser if the combined
        browser download is inconvenient. Reader-owned downloads are the explicit storage
        exception for this portable notebook; no B2 account or Modal Volume is required.

        The [visualization notebook]({visualisation_url}) displays the immutable archived
        reference. Keep this reader run separate, especially after a shorter duration,
        a radius subset or a different observed failure outcome.
    ''')
    cell('code', ''.join(common[6]['source']))
    result = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': dict(display_name='Python 3', language='python', name='python3'),
        'language_info': dict(name='python'), 'colab': dict(provenance=[]),
        'gc2d': dict(study_prefix=protocol['study_prefix'], reference_run_id=protocol['reference_run_id'],
            kit_url=KIT_URL, kit_sha256=manifest['sha256'], manifest_sha256=sha256(manifest_bytes).hexdigest(),
            protocol_sha256=sha256(protocol_bytes).hexdigest(), visualisation_url=visualisation_url)})
    target = ROOT / 'calcul.ipynb'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visualisation-url', default=VISUALISATION_URL)
    print(build(parser.parse_args().visualisation_url))
