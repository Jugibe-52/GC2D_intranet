# Poincare particle selector

Open `poincare_selector.html` in a browser. It works offline and contains all 5,000 returns for each of the 48 particles, with their original colors. Check particle boxes to show or hide them, choose **Only** to isolate one, or use **Show all**, **Hide all** and **Invert**. Point size and opacity are adjustable. The PNG button exports the selected plot.

`visualizacion.ipynb` reproduces this HTML from the existing sibling study `../Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot`, run `aws_48p_5000c_20s_16proc_spot_20260920`. It verifies the source CSV and metadata checksums, uses all 240,000 returns with no subsampling, excludes cycle zero and uses wrapped coordinates x/L and y/L. No new numerical calculation, reference or AWS instance is needed. Run the notebook using the GC2D project environment, with this folder as the working directory. The reusable exporter is `src/visualization/poincare_selector.py` in the project.
