# Notebook studies

- For new or revised numerical studies, keep two notebooks together in one study folder: one for calculation and one for visualization.
- The calculation notebook runs the study and persists its results, individual timings, parameters and provenance. The visualization notebook loads those saved results so changes to plots do not repeat trajectory integrations.
- Store scientific results in the project's configured Backblaze results bucket by default: `gc2d_data:gc2d-notebooks-data`. Use the existing persistence facilities in [diagnostics.io.persistence](../src/diagnostics/io/persistence.py) and [diagnostics.io.paths](../src/diagnostics/io/paths.py) where applicable. Both notebooks must share an explicit study prefix and run identifier.
- Local temporary files, download caches and recovery copies may support this workflow; a local `results/` folder must not silently replace the bucket as the final destination. Report failed uploads as failures and preserve recoverable data. Keep result storage separate from notebook-source synchronization.
- In runtime comparisons, exclude result upload and download from the measured numerical execution time unless transfer performance is itself the subject of the study; identify what each timing includes.
- A concrete scientific or technical reason, or an explicit user requirement, may justify a different organization or storage destination. Explain and document that reason in the study rather than silently changing the default.
- Apply this preference within authorized work. It does not itself authorize creating study notebooks, running simulations, migrating existing results or changing credentials.
