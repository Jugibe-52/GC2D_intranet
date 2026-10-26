# Time rows in States

States.states and States.values use (samples, state_size), including construction, recording and selection. Each row retains the existing component-major physical vector. Single states and numerical coordinate operations are unchanged.

Component accessors retain (particles, samples) for vectorized consumers, adapting storage explicitly. Specialized study references and archive formats retain their documented component-first contracts with conversions at States boundaries.

New save_solution archives use schema 2 and time rows. load_solution transposes schema 1 histories, including square matrices, based on version rather than shape inference. Existing files are not rewritten.

Completed States pickles include a history version; older restoration payloads decode their component columns explicitly.
