# Readable input preparation

Separate substantial input checks from object construction and plotting so that
the main operation is easy to follow. The repository-wide review found useful
extractions in physical state ownership, implicit method controls, comparison
records and study configuration. Ten plotting entry points also create figures
before discovering invalid later series.

Keep established accepted inputs, conversion order, error types and messages,
array ownership, numerical algorithms and public imports. Invalid plotting data
must be rejected before a figure is created.
