# Affected implementation and uses

The extended composition, trace records, GC elementary maps, analytic derivatives, JAX kernel and ABBA/BM4 class hierarchy change together. Keep one canonical implementation and explicit public exports through methods.extended, methods and simulation. Existing generic direct/adjoint formulations remain available, but the new composition engine executes explicit operations.

Preserve existing constructor options, solver choices, observer record shapes, stage counts and passive-energy conventions for named presets. New generic methods publish generic physical observations and operation metadata. Tests must distinguish elementary-operation counts from historical grouped stages, and verify signed time, reversibility, convergence, derivatives, independent particle packing, CPU/JAX agreement and callback behavior.

Update shared extended-model documentation and affected ABBA/BM4 model entry points, theoretical PDFs, current contracts and Spanish reading copies. Add a small runnable example for center/end time placement. Existing studies and notebooks retain their named methods; use a reduced representative notebook check rather than long campaign runs. Do not modify saved scientific results or the completed uncoupled-BM4 change record.
