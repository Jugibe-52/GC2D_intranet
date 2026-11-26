"""Rebuild the detailed-view notebook without running the figure scripts."""
from pathlib import Path

import nbformat


def build_notebook():
    cells = [
        nbformat.v4.new_markdown_cell("""# Trajectory similarity: detailed views

This notebook gathers the particle 33 and particle 12 figures from section 4.7.
It reads the saved figures from campaign `main-20261117-rho-div20`, at normalized radius `0.015`.
The plotting scripts use archived trajectories and do not run integrations.
The full matrices and group averages remain in [visualisation.ipynb](visualisation.ipynb)."""),
        nbformat.v4.new_code_cell('''from pathlib import Path
import json
import subprocess
import sys
from IPython.display import Image, display

candidates = (Path.cwd().resolve(), *Path.cwd().resolve().parents)
ROOT = next(p for base in candidates
            for p in (base, base / "TFM_redaccion", base.parent / "TFM_redaccion")
            if (p / "datos/campana_referencia.json").is_file())
reference = json.loads((ROOT / "datos/campana_referencia.json").read_text())
assert reference["representative_rho"] == 0.015
REGENERER_FIGURES = False
views = (
    ("Transport de la particule 33", "particula33_transporte.py", "transporte_particula33_fr"),
    ("Concordance de la particule 12", "particula12_concordancia.py", "concordancia_particula12_fr"),
    ("Matrices de la particule 12", "particula12_matrices.py", "semejanza_particula12_fr"),
)
'''),
        nbformat.v4.new_markdown_cell("""## Saved figures

Enable `REGENERER_FIGURES` only to export figures again from existing data.
The source scripts verify the archive fingerprints before export.
The semi-confinement view is displayed separately from its saved image."""),
        nbformat.v4.new_code_cell('''for title, script, stem in views:
    if REGENERER_FIGURES:
        subprocess.run([sys.executable, "-B", str(ROOT / "notebooks" / script)],
                       cwd=ROOT, check=True)
    provenance = ROOT / "figuras" / (stem + ".json")
    record = json.loads(provenance.read_text())
    assert record["campaign_id"] == reference["campaign_id"]
    assert record["rho"] == reference["representative_rho"]
    print(title)
    display(Image(filename=str(ROOT / "figuras" / (stem + ".png"))))

print("Semi-confinement de la particule 12")
display(Image(filename=str(ROOT / "figuras/semiconfinamiento_particula12_fr.png")))
'''),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"details-{index:02d}"
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (GC2D)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbformat.validate(notebook)
    return notebook


if __name__ == "__main__":
    destination = Path(__file__).with_name("visualisation_details.ipynb")
    nbformat.write(build_notebook(), destination)
    print(destination)
