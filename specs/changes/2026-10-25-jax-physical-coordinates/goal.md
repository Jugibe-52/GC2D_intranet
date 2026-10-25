# Explicit array backends for physical coordinates

Separate NumPy and JAX array operations while retaining one implementation of
the GC and FC equations. Currently coordinates and dynamics inspect their
inputs to select an array library, while execution substitutes JAX potentials.
Make that choice explicit through coordinate classes and small dynamics subclasses.

Keep physical component order, normalization, parameters, equations, saved
states and numerical methods unchanged. JAX remains optional for NumPy use.

The array module is a fixed class attribute, selected once when its class is defined.
