"""Render the package overview and execution flow from one diagram definition.

Write matching PlantUML/Graphviz, SVG and PNG artifacts without requiring a
Graphviz executable. Positions are in figure inches, with the origin at the
bottom left; the DOT source preserves them using Graphviz's neato layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/simulation/package-workflow-architecture"
INK = "#263D59"
BLUE = "#E8EFF8"
GREEN = "#E4F1E9"
ORANGE = "#FBE8D3"
PURPLE = "#EEE7F6"


@dataclass(frozen=True)
class Card:
    """One responsibility or operation, shared by all output formats."""

    key: str
    x: float
    y: float
    width: float
    height: float
    title: str
    owner: str
    body: str
    color: str = BLUE


PACKAGES = (
    ("potential", "Physical field", "Potential; load_gc2d_h5_potential\nGrid, interpolation, derivatives\nUnits and normalization\n\nConsumed by physical dynamics", BLUE),
    ("dynamics", "Equations of motion", "GuidingCenterDynamics\nFullCyclotronDynamics\nVector field and capabilities\n\nInput to InitialValueProblem", BLUE),
    ("initial_conditions", "Initial geometry", "GC / FC initial configurations\nPacked initial state and layout\nParticle count and geometry\n\nInput to InitialValueProblem", BLUE),
    ("formulations", "Numerical coordinates", "Physical / DoubledFormulation\nOptional time and momentum\nGC direct / adjoint maps\n\nPrepared by each method", GREEN),
    ("methods", "One numerical step", "classical / adaptive / hbvm\nextended: ABBA and BM4\ninitialize; advance; observations\n\nShared nonlinear solvers", ORANGE),
    ("integration", "Common run lifecycle", "IntegrationMethod.new_run\nControllers and output sampling\nCollector; integrate_method\n\nReturns IntegrationData", BLUE),
    ("simulation", "Public execution facade", "simulate(problem, method, request)\nCalls method.integrate\nChecks consistency with inputs\n\nBuilds and returns Solution", BLUE),
    ("solution.py", "Physical result module", "Solution: immutable output\nt: (T,); states: (m, T)\nSource layout and diagnostics\n\nRequested physical samples", PURPLE),
)

CARDS = [
    Card(f"package_{index}", 1.9 + 3.2 * index, 11.65, 2.98, 3.0,
         f"{index + 1:02d}  {name}",
         f"src/{name}" + ("" if name.endswith(".py") else "/"),
         f"{role}\n\n{body}", color)
    for index, (name, role, body, color) in enumerate(PACKAGES)
]
CARDS += [
    Card("contracts", 13.1, 9.1, 25.38, 1.35, "contracts/ | Shared records and interfaces", "src/contracts/",
         "InitialValueProblem = dynamics + initial_configuration     |     SimulationRequest = time span + step bound + saved times\n"
         "StateLayout / InitialConfiguration     |     StepInfo / StepResult     |     IntegrationData     |     Observation records"),
    Card("entry", 2.4, 6.3, 3.8, 2.0, "1  simulate", "simulation/runner.py",
         "Inputs: problem, method, request\nValidate entry-point types\nCall method.integrate"),
    Card("prepare", 6.68, 6.3, 3.8, 2.0, "2  Prepare a fresh run", "integration/core.py + methods/",
         "new_run calls method.initialize\nBind formulation and resources\nFreeze initial state and metadata"),
    Card("coordinate", 10.96, 6.3, 3.8, 2.0, "3  Coordinate integration", "integration/core.py",
         "integrate_method + controller\nAccepted intervals and saved samples\nCollect work; dispatch observations"),
    Card("step", 15.24, 6.3, 3.8, 2.0, "4  Advance one step", "methods/ + selected controller",
         "controller.steps calls advance\nEvaluate the method's numerical map\nReturn StepInfo and StepResult", ORANGE),
    Card("export", 19.52, 6.3, 3.8, 2.0, "5  Export collected history", "integration/ + formulations/",
         "After the accepted-step loop\nexport_history extracts physical states\nReturn IntegrationData to simulate", GREEN),
    Card("result", 23.8, 6.3, 3.8, 2.0, "6  Return the solution", "simulation/runner.py + solution.py",
         "simulate constructs Solution\nCheck saved times and initial state\nReturn immutable physical samples", PURPLE),
    Card("extended", 6.7, 2.65, 6.2, 1.75, "methods/extended/ | ABBA and BM4", "abba.py / bm4.py + core/",
         "Palindromic composition of signed direct / adjoint stages\nOne outer projection: implicit solve or midpoint average\nShared composition, Newton / Broyden and passive energy", ORANGE),
    Card("maps", 15.2, 2.65, 6.2, 1.75, "formulations/ | State and split maps", "state.py + gc.py",
         "DoubledFormulation: two spatial copies; optional energy\nGCDoubledMaps: direct_map / adjoint_map\nEvaluate shears and optional harmonic coupling", GREEN),
    Card("physics", 23.0, 2.65, 5.8, 1.75, "dynamics/ + potential/ | Physical evaluations", "dynamics/gc.py + potential/",
         "Guiding-center field from the configured potential\nHamiltonian and time derivative when tracking energy\nAnalytic spatial derivatives when the solver needs them"),
]
BY_KEY = {card.key: card for card in CARDS}

# The upper cards are an ordered ownership map. Only execution/detail nodes
# carry arrows; adjacent source packages need not call each other.
EDGES = (
    ("entry", "prepare"), ("prepare", "coordinate"),
    ("coordinate", "step"), ("step", "export"), ("export", "result"),
    ("extended", "maps"), ("maps", "physics"),
)
NOTES = (
    (0.4, 14.15, "GC2D | Package responsibilities and execution workflow", 22),
    (0.4, 13.62, "A  SOURCE ORGANIZATION   Left to right: responsibility order; each card identifies its implementation owner.", 12),
    (0.4, 7.98, "B  EXECUTION   Start in simulation; prepare the method; integrate; return a physical Solution.", 12),
    (13.1, 7.52, "Accepted-step loop: repeat 3-4, sample outputs and collect statistics", 10),
    (0.4, 4.1, "C  EXTENDED-FAMILY DETAIL   Numerical operations inside advance for ABBA and BM4", 12),
    (0.4, 1.13, "Assembly: potential feeds dynamics; dynamics and initial_conditions join in InitialValueProblem. The method prepares its formulation.", 11),
    (0.4, 0.65, "Surrounding workflow: studies/ assembles runs; diagnostics/ handles opt-in analysis and persistence; visualization/ presents results.", 11),
    (0.4, 0.22, "Notation: m = packed physical state size; T = requested saved times. Internal doubled or energy coordinates are exported through the formulation.", 10),
)


def render() -> None:
    """Create the three documentation artifacts from the cards and edges."""
    fig, ax = plt.subplots(figsize=(26.2, 14.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set(xlim=(0, 26.2), ylim=(0, 14.6))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for x, y, note, size in NOTES:
        ax.text(x, y, note, fontsize=size, color=INK,
                weight="bold" if size >= 12 else "normal",
                ha="center" if y == 7.52 else "left", va="center")

    for source, target in EDGES:
        a, b = BY_KEY[source], BY_KEY[target]
        ax.add_patch(FancyArrowPatch(
            (a.x + a.width / 2 + .03, a.y), (b.x - b.width / 2 - .03, b.y),
            arrowstyle="-|>", mutation_scale=14, color="#64748B", linewidth=1.5,
        ))
    # Show the repeated controller/method interaction above the execution row.
    ax.add_patch(FancyArrowPatch(
        (14.5, 7.34), (11.7, 7.34), arrowstyle="-|>", mutation_scale=12,
        linestyle="--", color="#4676A7", linewidth=1.2,
    ))
    ax.add_patch(FancyArrowPatch(
        (15.24, 5.25), (8.2, 3.56), arrowstyle="-|>", mutation_scale=13,
        linestyle=":", color="#B18450", linewidth=1.3,
    ))

    for card in CARDS:
        left, bottom = card.x - card.width / 2, card.y - card.height / 2
        ax.add_patch(FancyBboxPatch(
            (left, bottom), card.width, card.height,
            boxstyle="round,pad=0.015,rounding_size=0.09",
            facecolor=card.color, edgecolor="#C4CFDC", linewidth=1.0,
        ))
        ax.text(left + .14, bottom + card.height - .24, card.title,
                fontsize=11.1, weight="bold", color=INK, va="center")
        ax.text(left + .14, bottom + card.height - .54, card.owner,
                fontsize=9, color="#62748C", va="center")
        ax.text(left + .14, bottom + card.height - .80, card.body,
                fontsize=9.7, color="#34495E", va="top", linespacing=1.45)

    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "gc2d-packages"}):
        fig.savefig(OUTPUT.with_suffix(".svg"), metadata={"Date": None})
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=130)
    plt.close(fig)
    write_source()


def write_source() -> None:
    """Write PlantUML-compatible DOT with the same content and pinned layout."""
    source = ["@startdot", "digraph PackageWorkflow {",
              '  graph [layout=neato, overlap=true, splines=polyline, bgcolor="white"];',
              '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=11, pin=true];',
              '  edge [color="#64748B", penwidth=1.5, arrowsize=0.7];']
    for card in CARDS:
        body = escape(card.body).replace("\n", '<BR ALIGN="LEFT"/>')
        label = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="5">'
            f'<TR><TD ALIGN="LEFT"><B>{escape(card.title)}</B></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="#62748C">{escape(card.owner)}</FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">{body}</FONT></TD></TR></TABLE>'
        )
        source.append(
            f'  {card.key} [pos="{card.x},{card.y}!", '
            f'width={card.width}, height={card.height}, fillcolor="{card.color}", '
            f'color="#C4CFDC", label=<{label}>];'
        )
    for index, (_, y, note, size) in enumerate(NOTES):
        # Full-width labels preserve the figure's alignment in pinned DOT.
        align = "CENTER" if y == 7.52 else "LEFT"
        label = f'<TABLE BORDER="0" WIDTH="1829"><TR><TD ALIGN="{align}">{escape(note)}</TD></TR></TABLE>'
        source.append(f'  note_{index} [shape=plaintext, style="", pos="13.1,{y}!", fontsize={size}, label=<{label}>];')
    source.extend(f"  {a} -> {b};" for a, b in EDGES)
    source += ['  step -> coordinate [style=dashed, color="#4676A7"];',
               '  step -> extended [style=dotted, color="#B18450"];', "}", "@enddot", ""]
    OUTPUT.with_suffix(".puml").write_text("\n".join(source))


if __name__ == "__main__":
    render()
