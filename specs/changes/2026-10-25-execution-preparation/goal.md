# Clear execution preparation

`Execution.run` mixes the complete simulation lifecycle with backend, observer,
method and compiled-step preparation. Readers must follow those details before
they can see how a run starts, records its result and finishes after success or
failure.

Make that lifecycle easy to follow while preserving numerical results, public
calls, diagnostics, validation and optional JAX loading.
