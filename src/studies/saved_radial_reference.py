"""Select a single saved radial orbit for planar energy diagnostics."""

from copy import deepcopy
from dataclasses import replace
import numpy as np
from diagnostics import StoredReferenceTrajectory


def select_reference_particle(reference: StoredReferenceTrajectory, particle: int) -> StoredReferenceTrajectory:
    """Extract one zero-based particle without resampling; preserve parent provenance."""
    count = reference.initial_state.size // 2
    if not isinstance(particle, int) or not 0 <= particle < count:
        raise ValueError("Particle index is outside the saved reference.")
    rows = [particle, count + particle]
    metadata = deepcopy(dict(reference.metadata))
    metadata["parent_trajectory_sha256"] = metadata.pop("trajectory_sha256", None)
    metadata["selected_particle_index"] = particle
    metadata["particle_count"] = 1
    metadata["state_layout"] = "component_major_[x_1,y_1]"
    metadata["selection"] = "In-memory particle selection; paths identify the parent artifact."
    distances = reference.audit_distances[particle:particle+1]
    states, audit = reference.states[rows], reference.audit_states[rows]
    metadata["audit"] = {
        "global_rms_distance": float(np.sqrt(np.mean(distances**2))),
        "maximum_distance": float(distances.max()),
        "final_rms_distance": float(distances[0,-1]),
        "final_maximum_distance": float(distances[0,-1]),
        "maximum_state_component_difference": float(np.max(np.abs(states-audit))),
    }
    metadata["audit_maximum_distance_per_particle"] = [float(distances.max())]
    metadata["parent_initial_conditions"] = metadata.pop("initial_conditions")
    metadata["initial_conditions"] = {"initial_state": reference.initial_state[rows].tolist(),
                                       "parent_particle_index": particle}
    return replace(reference, states=states, initial_state=reference.initial_state[rows],
                   audit_states=audit, audit_distances=distances, metadata=metadata)
