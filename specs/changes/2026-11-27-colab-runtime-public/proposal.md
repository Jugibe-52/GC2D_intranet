# Publish the saved runtime visualization

The user accepted this scope on 2026-11-27 after the public-bucket and Colab approach had been presented. Adapt the section 4.2 visualization notebook to download `timings.json`, `study.json` and a publication manifest over public HTTPS from a new B2 bucket. Keep the original private study at `studies/method_timings/jax-20261118-rho0015-c500` unchanged. The public manifest describes only the selected files and their verified checksums and provenance.

The notebook verifies the downloaded bytes, derives the first-call values and cached medians and minimum–maximum ranges from the original 32 timing records, and reproduces the existing comparison and ratios to RK2. It uses the recorded 40-particle, 500-cycle, rho=0.015 study; it does not rerun integrations or reinterpret these times as accuracy comparisons. Keep notebook prose in English and figure text in French.

Publish only this visualization notebook in `COLAB NOTEBOOKS/MEMOIRE_TFM/04_02_cout_calcul/visualisation.ipynb`, preserving its Drive identity on later updates. Store the figure output for immediate viewing. Public B2 files need no Colab secrets; publication credentials remain outside notebooks and public files. Record the actual public URLs and Drive/Colab link only after the corresponding operations have succeeded.
