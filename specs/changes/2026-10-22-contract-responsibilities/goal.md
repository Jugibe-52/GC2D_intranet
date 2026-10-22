# Clear ownership around contracts

Keep shared interfaces and data records in `contracts`. Place concrete problem
construction, execution configuration, study policy and diagnostic calculations
with their owners. The existing array helpers apply the shared NumPy/JAX
compatibility agreement and do not need a generic utilities package.

Physical equations, units, state ordering, numerical algorithms, tolerances and
saved-data formats must retain their current meaning and behavior.
