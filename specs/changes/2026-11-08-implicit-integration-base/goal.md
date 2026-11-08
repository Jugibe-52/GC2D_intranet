# Separate implicit method preparation

Every IntegrationMethod currently accepts and stores a field Jacobian, even
when its explicit stages never use it. Execution and JAX assume that all methods
have this dependency. Make the common contract describe every integrator, and
give algebraic preparation its own ImplicitIntegrationMethod base.

Preserve numerical equations, solver defaults, physical coordinates and saved
results. This change concerns ownership and preparation, not numerical accuracy.
