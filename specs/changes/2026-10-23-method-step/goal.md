# Method-owned step size

Make the integration step a numerical-method setting, independent of the time
span and output samples. A fixed method uses its configured positive `step`
throughout a run and shortens only the final accepted step to reach the endpoint.
Currently `SimulationRequest.max_step` instead creates a uniformly rescaled grid.
