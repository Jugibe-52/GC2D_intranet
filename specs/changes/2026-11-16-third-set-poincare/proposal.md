# Calculation and output

Use the existing BM4Implicit method and reflected measured potential, rho = 0.30, B = 1.5 T, 5000 physical forcing cycles and 40 fixed steps per cycle. Reuse the campaign loader and method factory: constant mode 0, variable mode 15, cubic interpolation, no smoothing, Newton absolute/relative tolerances 1e-12/1e-11 and at most 40 corrections. These defaults remain editable.

The notebook saves unwrapped normalized coordinates at whole cycles, preserving particle IDs and colors. Its plot uses small colored points, equal axes and an initial [0,1] square viewport; report any points outside that viewport without wrapping them. Export French-labelled PNG/PDF figures and a compressed numerical archive with exact inputs, source fingerprints and validation results. Plot appearance alone does not establish chaos or convergence over long times.
