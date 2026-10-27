# Separate compilation from execution

Fixed-step JAX `run_steps` currently builds the schedule, places inputs, and
triggers compilation when it first calls the integrator. Make preparation and
compilation explicit before this worker starts, so its responsibility is to
execute an already compiled integration and collect its results.
