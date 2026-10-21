# Proposed behavior

Remove coupling_frequency from both BM4 constructors. Bind their CPU and JAX maps without coupling, preserving the twelve signed coefficients, endpoint clock, outer implicit projection or final arithmetic mean, solver controls, and passive physical momentum normalization.

Expose HarmonicCoupling as an independent operation on two packed planar copies. Reuse the existing exact coupling matrix and keep the generic coupled formulation available. BM4 does not invoke C, even as a zero-frequency identity. The general A/B/T inheritance redesign remains a separate design discussion.

Existing study schemas and observer/archive records may retain coupling_frequency as zero-valued metadata. Current calculations must reject nonzero legacy requests before integration; archived coupled results are not relabeled or overwritten. New BM4 defaults and migrated examples use no coupling. The midpoint default changes from pi/8 to no coupling; the implicit zero-coupling default retains its numerical behavior.
