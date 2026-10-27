# Shared fixed-step planning

Centralize the main grid and output-sample decisions used by host and JAX execution. Build the plan before backend-specific preparation so `prepare_integration` only places inputs and obtains the executable. Preserve numerical results and the existing local compilation changes.
