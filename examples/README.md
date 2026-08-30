# Supported examples

These small scripts exercise the flattened public packages without depending
on local development notebooks. They are intended to remain fast enough for
smoke testing and documentation review.

- `gc_orbit.py` runs a short guiding-center RK4 trajectory.
- `projected_abba.py` runs one short `ABBA2Implicit` trajectory with the
  `reduced_multiplier` formulation and prints its nonlinear-solver diagnostics.

Run either script from the project root after installing the editable project
environment with `python -m pip install -r requirements.txt`.
