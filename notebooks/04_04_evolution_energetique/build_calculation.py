"""Build the self-contained reader calculation for the current section 4.4."""
from hashlib import sha256
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT.parents[1] / 'outputs/colab-public/04_04_evolution_energetique/calculation/v1'
KIT_URL = 'https://f003.backblazeb2.com/file/gc2d-colab-public/04_04_evolution_energetique/calculation/v1/manifest.json'
VISUALISATION_URL = 'https://colab.research.google.com/drive/14kK36hblOqiPnGgDMMTO8KId7gXwd7dH'


def build(visualisation_url=VISUALISATION_URL):
    manifest_bytes = (PUBLIC / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    common = json.loads((ROOT.parent / '04_03_numpy_jax/calcul.ipynb').read_text())
    old = [c for c in common['cells'] if c['cell_type'] == 'code']
    cells = []
    def cell(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        value = dict(cell_type=kind, id=sha256(source.encode()).hexdigest()[:12],
            metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code': value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell('markdown', f'''
        # 4.4 — Calculate the online energy balance

        Open this notebook in **Google Colab with a standard CPU runtime**. It runs the
        eight methods on **your personal Modal account**, with no local GC2D checkout
        or B2 credentials. Public B2 supplies verified frozen inputs; your results are
        offered as a **ZIP download to your computer**.

        The reference is `main-20261117-rho-div20-online-balance-20261120`: 84 particles,
        normalized rho=0.015, 5000 cycles, 40 fixed steps and ten saved positions per cycle.
        Each method runs in a new process, using the original numerical sources, physical
        potential, initial positions and exact archived time plan. The trajectory evolves
        freely. Agreement with the main campaign has not been established.

        Two passive accumulators integrate the physical partial time derivative of H:
        Simpson on two steps, and Simpson on four steps. At the end of each cycle,
        R2 = H-H0-I2 and R4 = H-H0-I4. The output reports the RMS of each residual and
        RMS(R4)/RMS(R2) for the eight periodic candidates and eight chaotic particles.
        These are the diagnostics retained in section 4.4. The ratio measures sensitivity
        to quadrature; it is neither an error bound nor a trajectory-accuracy certificate.

        The historical calculation program is kept byte-for-byte as provenance. Its raw
        arrays retain H and H-H0 because both residuals require them; this notebook does
        not reproduce the removed Hamiltonian-variation figures or summaries.
        The [visualization notebook]({visualisation_url}) displays the saved reference
        with the four energy-balance panels, without calculating trajectories again.

        Create a [Modal account](https://modal.com/) and a personal API token.
        Remote computation uses your account's allowance and billing; check
        [current pricing](https://modal.com/pricing) before starting.
        The worker requests two CPU cores and 4 GiB memory (12 GiB limit), with a four-hour
        timeout and no automatic retry. The eight methods run sequentially.
        Full results contain approximately 0.65 GB of arrays before ZIP packaging;
        download them promptly because Modal call-result retention and Colab storage are temporary.
    ''')
    cell('markdown', '## 1. Install the client\nThe numerical dependencies are installed in the remote worker. NumPy checks the returned arrays here.')
    cell('code', ''.join(old[0]['source']))
    cell('markdown', '''
        ## 2. Authenticate with your personal Modal account

        Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to Colab's **Secrets** panel
        and grant this notebook access. Hidden prompts provide a fallback.
        Do not paste credentials into cells. They stay in this runtime's environment;
        the notebook does not save them in outputs or result files.
    ''')
    cell('code', ''.join(old[1]['source']))
    cell('markdown', '''
        ## 3. Set the duration and explicitly enable launch

        The reference uses `CYCLES = 5000` and `BLOCK_CYCLES = 100`.
        For a quick workflow check, use `CYCLES = 2`, `BLOCK_CYCLES = 1`.
        All eight methods still integrate all 84 particles with the same fixed step.
        Reduced results are not the thesis reference.

        Set `LAUNCH_CALCULATION = True` when ready. Keep the generated `RUN_ID` and
        download the receipt. After a disconnected session, restore that identifier,
        the same settings and `RESUME_CALL_ID` to retrieve the existing Modal call.
        A new run needs a new identifier. The worker has local block checkpoints while
        it lives; this notebook does not promise checkpoint survival after its termination.
    ''')
    cell('code', '''
        from datetime import datetime, timezone
        from pathlib import Path
        from uuid import uuid4

        CYCLES = 5000  # @param {type:"integer"}
        BLOCK_CYCLES = 100  # @param {type:"integer"}
        LAUNCH_CALCULATION = False  # @param {type:"boolean"}
        RUN_ID = globals().get("RUN_ID", "reader-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
        RESUME_CALL_ID = ""  # @param {type:"string"}
        WORKSPACE = Path.cwd() / "gc2d-colab-44"
        WORKSPACE.mkdir(exist_ok=True)
        print(f"Run: {RUN_ID}; eight methods, {CYCLES} cycles, {CYCLES * 40} steps per method.")
    ''')
    cell('markdown', f'''
        ## 4. Download and verify the frozen reproduction kit

        The download is approximately **{manifest['bytes'] / 1e6:.0f} MB**, split into
        three public objects. The full physical HDF5 file is kept byte-identical because
        the historical code verifies its fingerprint. The pinned manifest, each part,
        the assembled archive and every extracted input are checked before execution.
        Numerical code and time plans are unchanged. Only the author's source filename
        in descriptive input metadata is anonymized; original and public hashes are recorded.
        Embedded helper code makes the notebook independent of a repository checkout.
    ''')
    runtime = (ROOT / 'calculation_runtime.py').read_text()
    worker = (ROOT / 'calculation_worker.py').read_text()
    cell('code', f'''
import importlib
import json
import sys

KIT_URL = {KIT_URL!r}
MANIFEST_SHA256 = {sha256(manifest_bytes).hexdigest()!r}
KIT_SHA256 = {manifest['sha256']!r}
(WORKSPACE / "calculation_runtime.py").write_text({runtime!r})
(WORKSPACE / "calculation_worker.py").write_text({worker!r})
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
importlib.invalidate_caches()
import calculation_runtime as runtime
runtime = importlib.reload(runtime)
settings = runtime.check_settings(CYCLES, BLOCK_CYCLES, RUN_ID)
inputs = runtime.download_kit(KIT_URL, MANIFEST_SHA256, KIT_SHA256, WORKSPACE / "kit")
import calculation_worker as worker
worker = importlib.reload(worker)
print("Original scientific inputs and numerical sources verified. No calculation has started.")
''')
    cell('markdown', '''
        ## 5. Launch once, or recover the recorded call

        The receipt is written before waiting. Repeating this cell reuses a confirmed call.
        An interrupted, ambiguous submission never triggers an automatic replacement:
        inspect your [Modal dashboard](https://modal.com/apps), then enter the existing call ID
        as `RESUME_CALL_ID` to recover it without submitting another calculation.
    ''')
    cell('code', ''.join(old[4]['source']))
    cell('markdown', '''
        ## 6. Retrieve and verify the result

        This cell waits on the recorded call and never starts another integration.
        Retry the same call after a wait timeout. If its result has expired, check any ZIP
        already downloaded. The archive contains all trajectories and cycle diagnostics,
        individual timings, parameters, frozen source and a complete checksum manifest.
        Reported numerical timings include compilation and online diagnostics, and exclude
        input restoration, checks, checkpoints, compression and network transfers.
    ''')
    cell('code', '''
        RESULT_ZIP = RUN_DIRECTORY / (RUN_ID + ".zip")
        if RESULT_ZIP.exists():
            result_bytes = RESULT_ZIP.read_bytes()
        else:
            result_bytes = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=3600)
        study, summary = runtime.verify_results(result_bytes, settings, inputs)
        temporary_zip = RESULT_ZIP.with_suffix(".tmp")
        temporary_zip.write_bytes(result_bytes)
        temporary_zip.replace(RESULT_ZIP)
        print(f"Eight methods verified; {CYCLES} cycles each; ZIP: {len(result_bytes):,} bytes.")
        print(f"{'Method':20} {'Group':22} {'RMS(R2)':>13} {'RMS(R4)':>13} {'RMS4/RMS2':>13}")
        for method, groups in summary.items():
            for group, values in groups.items():
                ratio = values["ratio_4_over_2"]
                ratio_text = f"{ratio:.6g}" if ratio is not None else "undefined"
                print(f"{method:20} {group:22} {values['rms_simpson_2']:13.6e} {values['rms_simpson_4']:13.6e} {ratio_text:>13}")
    ''')
    cell('markdown', '''
        ## 7. Save the results on your computer

        Allow the download and keep the ZIP and receipt before closing the runtime.
        Saving the notebook does not save these files. No B2 account or Modal Volume is
        needed: reader-owned downloads are the deliberate destination for new results.
        The separate visualization notebook displays the saved 5000-cycle reference.
    ''')
    cell('code', ''.join(old[6]['source']))
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': dict(display_name='Python 3', language='python', name='python3'),
        'language_info': dict(name='python'), 'colab': dict(provenance=[]),
        'gc2d': dict(kit_url=KIT_URL, kit_sha256=manifest['sha256'],
            manifest_sha256=sha256(manifest_bytes).hexdigest(), reference_run_id=manifest['reference_run_id'])})
    destination = ROOT / 'calcul.ipynb'
    backup = ROOT.parents[1] / 'outputs/colab-backups/04_04_evolution_energetique/calcul-before-colab.ipynb'
    if destination.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(destination.read_bytes())
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visualisation-url', default=VISUALISATION_URL)
    print(build(parser.parse_args().visualisation_url))
