# Diagnostics grouped by the observed process

Make each diagnostic easy to locate beside the responsibility it observes.
The current flat package mixes numerical steps, physical-state geometry,
execution progress, study analysis and persistence. Its organization should
follow the corresponding parts of `src` while keeping shared calculations in
one place.

This is an organization change: numerical formulas, observation timing,
solver tolerances, array layouts and saved artifact formats remain unchanged.
