"""Section 4.1: inspect the reference and prepare a separate new campaign."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


SOURCE = Path(__file__).resolve()


def find_roots(start: Path | None = None) -> tuple[Path, Path]:
    """Find the sibling repositories from a section, repository root or this script."""
    origin = (start or Path.cwd()).resolve()
    for base in (origin, *origin.parents, SOURCE.parent, *SOURCE.parents):
        if (base / "src/studies/main_campaign.py").is_file():
            code, thesis = base, base.parent / "TFM_redaccion"
        elif (base / "datos/campana_referencia.json").is_file():
            thesis, code = base, base.parent / "GC2D_codigo"
        else:
            continue
        if not (code / "src/studies/main_campaign.py").is_file():
            raise FileNotFoundError(f"GC2D_codigo repository not found beside {thesis}.")
        if not (thesis / "datos/campana_referencia.json").is_file():
            raise FileNotFoundError(f"Thesis reference selection not found beside {code}.")
        return thesis, code
    raise FileNotFoundError("Sibling GC2D_codigo and TFM_redaccion repositories not found.")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_selection(thesis: Path) -> dict:
    """The selection file is the single source of the thesis protocol."""
    selection = read_json(thesis / "datos/campana_referencia.json")
    if selection["rho_values"] != [i / 400 for i in range(11)]:
        raise ValueError("The selection no longer matches the rho = i/400 grid.")
    if selection["config"] != {"cycles": 5000, "steps_per_cycle": 40, "samples_per_cycle": 10}:
        raise ValueError("The saved protocol no longer matches 5000/40/10.")
    if (selection["particle_count"] != 84 or len(selection["methods"]) != 8
            or len(set(selection["methods"])) != 8 or selection["expected_cases"] != 88):
        raise ValueError("The protocol requires 84 particles and eight distinct methods.")
    return selection


def inspect_reference(thesis: Path, code: Path) -> dict:
    """Check records and local availability without loading the large trajectories."""
    selection = read_selection(thesis)
    directory = code / selection["campaign_path"]
    record = read_json(directory / "campaign.json")
    progress = read_json(directory / "progress.json")
    for saved, selected in (("config", "config"), ("rhos", "rho_values"),
                            ("methods", "methods"), ("source_sha256", "source_sha256"),
                            ("code_sha256", "code_sha256"), ("bucket", "bucket")):
        if record[saved] != selection[selected]:
            raise ValueError(f"Reference and selection disagree: {saved}.")
    particles = record["particles"]["particles"]
    if [p["id"] for p in particles] != list(range(1, 85)):
        raise ValueError("The historical IDs of the 84 particles have changed.")
    source_particles = read_json(thesis / "datos/conjuntos_particulas_rho030.json")
    if particles != source_particles["particles"]:
        raise ValueError("Particle positions, colors or IDs differ.")
    cases = progress["cases"]
    expected_pairs = {(rho, method) for rho in selection["rho_values"] for method in selection["methods"]}
    if (len(cases) != 88 or {(row["rho"], row["method"]) for row in cases.values()} != expected_pairs):
        raise ValueError("The record does not contain the 88 expected radius/method pairs.")
    complete = [key for key, row in cases.items() if row["status"] == "complete"]
    failed = sorted(key for key, row in cases.items() if row["status"] == "failed")
    if len(complete) != selection["available_cases"] or failed != sorted(selection["unavailable_cases"]):
        raise ValueError("Recorded availability differs from the expected 86 results and two failures.")
    local_missing = [key for key in complete if not (directory / "runs" / key / "manifest.json").is_file()]
    if any((directory / "runs" / key / "manifest.json").exists() for key in failed):
        raise ValueError("An unavailable case now has an archive: review the record.")
    return {
        "campaign": selection["campaign_id"],
        "preserved_numerical_status": progress["status"],
        "verified_results_according_to_record": len(complete),
        "local_manifests_present": len(complete) - len(local_missing),
        "local_manifests_missing": local_missing,
        "unavailable_cases": failed,
        "local_directory": str(directory),
        "results_destination": selection["bucket"],
        "visualization": selection["website_url"],
        "check": "Records and manifest availability; no new verification of B2 archive bytes.",
    }


def new_directory(code: Path, selection: dict, run_id: str) -> Path:
    """Reject the frozen reference, nested paths and their symbolic aliases."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("RUN_ID must be a simple name containing letters, digits, hyphens or underscores.")
    if run_id == selection["campaign_id"]:
        raise ValueError("The reference campaign is frozen: choose a different RUN_ID.")
    base = (code / "notebooks/developements/main_reflected_campaign").resolve()
    directory = (base / run_id).resolve()
    if directory.parent != base or directory == (code / selection["campaign_path"]).resolve():
        raise ValueError("The selected directory must not point to the reference or leave the campaigns folder.")
    return directory


def command_plan(thesis: Path, code: Path, run_id: str, *, workers: int = 4,
                 python: Path | None = None) -> list[list[str]]:
    """Build commands without running them or creating a directory."""
    selection = read_selection(thesis)
    directory = new_directory(code, selection, run_id)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Choose between one and sixteen workers.")
    executable = python or code / ".venv/bin/python"
    return [
        [str(executable), str(SOURCE), "--prepare", "--run-id", run_id],
        [str(executable), "-m", "modal", "deploy", "examples/modal_main_campaign_app.py"],
        [str(executable), "scripts/run_main_campaign.py", "run", str(directory), "--workers", str(workers)],
    ]


def prepare_new(thesis: Path, code: Path, run_id: str) -> None:
    """Prepare the explicit grid and methods without launching any worker."""
    selection = read_selection(thesis)
    directory = new_directory(code, selection, run_id)
    inspect_reference(thesis, code)
    sys.path.insert(0, str(code / "src"))
    from studies.main_campaign import CampaignConfig, METHODS, campaign_case_key
    from studies.main_campaign_io import prepare_campaign
    if tuple(selection["methods"]) != METHODS:
        raise ValueError("The method list in the code has changed: review the protocol before execution.")
    keys = {campaign_case_key(rho, method) for rho in selection["rho_values"] for method in selection["methods"]}
    if len(keys) != selection["expected_cases"]:
        raise ValueError("The case keys are not distinct.")
    prepare_campaign(directory, code, config=CampaignConfig(**selection["config"]),
                     particle_path=thesis / "datos/conjuntos_particulas_rho030.json",
                     rhos=tuple(selection["rho_values"]), methods=tuple(selection["methods"]))
    print(f"Local preparation complete: {directory / 'campaign.json'}")


def launch_new(thesis: Path, code: Path, run_id: str, *, execute: bool = False,
               workers: int = 4, python: Path | None = None) -> list[list[str]]:
    """Show the plan; execute=True prepares, deploys, calculates and archives on B2."""
    plan = command_plan(thesis, code, run_id, workers=workers, python=python)
    for command in plan:
        print(shlex.join(command))
    if not execute:
        print("Preview only: no writes, calculations or transfers.")
        return plan
    if "YYYYMMDD" in run_id or "a-renommer" in run_id:
        raise ValueError("Replace the example RUN_ID with an explicit, stable identifier.")
    inspect_reference(thesis, code)
    if not Path(plan[0][0]).is_file():
        raise FileNotFoundError("Repository interpreter missing; set PYTHON_GC2D to the prepared environment.")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for command in plan:
        subprocess.run(command, cwd=code, env=env, check=True)
    return plan


NOTEBOOK_CELLS = [('markdown', '# 4.1 — Main execution\n\nThis notebook checks the reference campaign and provides a calculation entry point for a separate new campaign. Visualization is available on the [reference website](https://gc2d-main-reflected-rho-div20.pages.dev/); this section therefore needs no visualization notebook.\n\nAll cells are read-only by default. The reference retains 86 results and two convergence failures, `ImplicitMidpoint` and `ABBA2Implicit` at $\\widetilde\\rho=0$.\n'), ('code', "from pathlib import Path\nimport runpy\nfrom types import SimpleNamespace\nimport json\n\n# Support the section directory and either sibling repository root.\nrelative_source = Path('notebooks/04_01_execution_principale/calcul.py')\ncandidates = [Path.cwd(), *Path.cwd().parents]\nsource = next((candidate for base in candidates\n               for candidate in (base / relative_source,\n                                 base.parent / 'GC2D_codigo' / relative_source)\n               if candidate.is_file()), None)\nif source is None:\n    raise FileNotFoundError('Cannot locate the section 4.1 calculation source.')\ncalcul = SimpleNamespace(**runpy.run_path(str(source)))\nTHESIS, CODE = calcul.find_roots()\nselection = calcul.read_selection(THESIS)\nprint(json.dumps(selection, ensure_ascii=False, indent=2))\n"), ('markdown', '## Reuse saved results\n\nThe grid is $\\widetilde\\rho_i=i/400$, $i=0,\\ldots,10$, in base-cell side units. No further division by 20 is applied. The 84 historical positions, IDs and colors are preserved; their `rho030` file records their provenance. Each case uses 5000 cycles, 40 fixed steps per cycle and ten saved samples per cycle. Coordinates remain unwrapped, with a spatial extension of period 2. All schemes evaluate the same physical Hamiltonian directly, without temporal quadrature.\n\nThe following check compares the selection with saved records and inventories local manifests. It does not repeat integrations, download data or repeat verification of B2 archives.\n'), ('code', 'etat = calcul.inspect_reference(THESIS, CODE)\nprint(json.dumps(etat, ensure_ascii=False, indent=2))\n'), ('markdown', '## Prepare a new run when needed\n\nThe frozen reference cannot be the target. Choose a distinct `RUN_ID` and retain it to resume the same Modal receipts. The preview shows commands for local preparation, deployment of the current worker and execution. Current source code may differ from the frozen reference code: its fingerprints are saved in the new campaign, which does not replace the thesis results.\n\nEnabling `EXECUTER_NOUVELLE_CAMPAGNE` runs the 88 cases on Modal/JAX CPU float64 and archives their results under `gc2d_data:gc2d-notebooks-data/developements/main_reflected_campaign/<RUN_ID>`. This requires a GC2D environment with its JAX/Modal dependencies, Modal authentication and the existing rclone configuration. Deployment updates the Modal application `gc2d-main-reflected`: use this cell only when no other campaign is active on that application. Local archives support recovery; they do not replace the B2 destination. Transfers are excluded from worker calculation timings. No website export or deployment is performed.\n\nNumerical and transfer failures remain explicit; never remove receipts to force resubmission. A resumed campaign uses the same identifier and unchanged sources.\n'), ('code', "RUN_ID = 'main-reproduction-YYYYMMDD'  # Replace before starting a new run.\nWORKERS = 4\nPYTHON_GC2D = CODE / '.venv/bin/python'\nEXECUTER_NOUVELLE_CAMPAGNE = False\n\nplan = calcul.launch_new(THESIS, CODE, RUN_ID,\n                          execute=EXECUTER_NOUVELLE_CAMPAGNE,\n                          workers=WORKERS, python=PYTHON_GC2D)\n")]

def write_notebook() -> Path:
    cells = []
    for index, (kind, source) in enumerate(NOTEBOOK_CELLS):
        cell = {"cell_type": kind, "id": f"execution-principale-{index}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    path = SOURCE.with_suffix(".ipynb")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true", help="Regenerate calcul.ipynb without numerical calculations.")
    parser.add_argument("--prepare", action="store_true", help="Prepare a new local campaign without submitting it.")
    parser.add_argument("--run-id", help="Stable identifier distinct from the reference.")
    args = parser.parse_args()
    if args.notebook and args.prepare:
        parser.error("Choose a single operation.")
    if args.notebook:
        print(write_notebook())
    else:
        thesis_root, code_root = find_roots()
        if args.prepare:
            if not args.run_id:
                parser.error("--prepare requires --run-id.")
            prepare_new(thesis_root, code_root, args.run_id)
        else:
            print(json.dumps(inspect_reference(thesis_root, code_root), ensure_ascii=False, indent=2))
