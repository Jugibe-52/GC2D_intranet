# Particle datasets

The viewing notebook [`particle_sets_potential.ipynb`](../../notebooks/particle_sets_potential.ipynb) displays these four datasets in order:

| Dataset | IDs | Purpose |
| --- | --- | --- |
| `original_40.json` | 1–40 | Original star distribution for an overview of the main Poincare structure. |
| `expanded_84.json` | 1–84 | The 40 star points, eight additional probes and 36 neighbors. |
| `chaotic_candidates.json` | 12, 15, 18, 21, 24, 27, 30, 33 | Selected candidates whose chaotic character is not verified for the reflected potential. |
| `periodic_candidates_8.json` | 1, 2, 60, 64, 65, 72, 76, 83 | Selected candidates whose periodicity is not verified for the reflected potential. |

The first dataset preserves the complete records of IDs 1–40 from `original_48.json`: physical positions, normalized coordinates, colors and identifiers. Initial positions form an eight-ray star centered at (0.5, 0.5), with radii increasing from zero to 0.425. The oval appearance refers to the historical Poincare returns, not to an ellipse used to place the initial points.

`original_48.json` remains the historical intermediate set used by earlier studies. It includes the star and the eight probes, but is no longer the first notebook panel. The 84-particle campaign and all saved trajectories keep their original inputs.

`chaotic_candidates.json` preserves eight complete records from `original_40.json` in the requested order. Its metadata records the user selection, without inheriting numerical observations from the former eight-particle selection. `chaotic_candidates_7.json` retains the preceding selection without particle 33; `chaotic_candidates_8.json` retains the older historical selection. Both remain outside the four active panels.

`periodic_candidates_8.json` preserves eight complete records from `expanded_84.json`: two original particles (1 and 2) and six added neighbors (60, 64, 65, 72, 76 and 83). Their original kinds and parent references remain intact. `periodic_8.json` retains the historical verified orbits with IDs 85–92 outside the active panels; its periods and closure checks do not apply to the new selection.

Each JSON records its source field and provenance. The notebook shows initial positions over the reflected, gyroaveraged potential without integrating trajectories. Its figure text is in French, following the thesis convention. The historical verified dataset retains the conditions and checks described in [Periodic particle selection](../../docs/studies/periodic-particles.md).

The notebook also displays a standalone view of the first 40 particles. Set `export_thesis_figure = True` to update `primer_conjunto_40_fr.pdf`, its PNG and the JSON parameter record in the existing sibling `TFM_redaccion/figuras/` directory. Export is disabled by default; ordinary viewing does not require the thesis checkout.
