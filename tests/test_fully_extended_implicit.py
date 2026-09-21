"""Migration safeguards for the retired time/momentum projection mode."""
import unittest
from simulation import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, simulate
from tests.test_method_integration import _problem, _request
from studies import run_fully_extended_implicit_study


class FormerFullProjectionTests(unittest.TestCase):
    def test_former_eight_component_mode_requires_explicit_migration(self):
        for cls in (ABBA2Implicit, ABBA4Implicit, ABBA6Implicit):
            with self.assertRaisesRegex(ValueError, "track_energy=True"):
                cls(state_extension="fully_extended")

    def test_energy_tracking_projects_only_space(self):
        for cls in (ABBA2Implicit, ABBA4Implicit, ABBA6Implicit):
            for formulation in ("reduced_multiplier", "simultaneous_state_multiplier"):
                events=[]
                result=simulate(_problem(), cls(track_energy=True, projection_formulation=formulation,
                    step_observer=events.append), _request())
                self.assertEqual(result.diagnostics["accepted_internal_state_dimension"],6)
                self.assertEqual(result.diagnostics["nonlinear_unknown_dimension"],
                    2 if formulation=="reduced_multiplier" else 6)
                self.assertEqual(events[0].state_before.shape,(2,))

    def test_historical_full_state_study_is_not_silently_reinterpreted(self):
        with self.assertRaisesRegex(NotImplementedError, "passive energy"):
            run_fully_extended_implicit_study(None,None,method="abba2_fully_extended_implicit",config=None)
