"""Regression coverage for reference reuse with a finer integration grid."""
import tempfile
import unittest
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from studies import RandomPotentialConfig, latin_hypercube_gc_configuration, FiveMethodComparisonConfig, run_five_method_comparison
from diagnostics import write_five_method_comparison_csv, load_five_method_comparison_csv
from visualization import plot_implicit_method_iterations

class ReusedThreeMethodTests(unittest.TestCase):
    def test_reuse_and_substep_csv(self):
        p=RandomPotentialConfig(amplitude=.2,max_wave_number=3,nx=16,ny=16,seed=27,interpolation_order=5).build()
        ic=latin_hypercube_gc_configuration(p,particle_count=3,seed=20260905,domain_margin_fraction=.35)
        c=FiveMethodComparisonConfig(t_span=(0.,.04),integration_step=.02,save_interval=.02,timing_warmups=0,timing_repeats=1,progress=False)
        a=run_five_method_comparison(p,ic,config=c)
        with patch('studies.five_method_sdirk_rk4_comparison.build_adaptive_reference',side_effect=AssertionError('Reference must not run')):
            r=run_five_method_comparison(p,ic,config=replace(c,integration_step=.005),method_names=('BM4Implicit','GaussLegendre4','RK4'),reused_reference=a.reference)
        with tempfile.TemporaryDirectory() as d:
         q=write_five_method_comparison_csv(r,Path(d)/'results.csv'); loaded=load_five_method_comparison_csv(q)
         for name in ('BM4Implicit','GaussLegendre4'):
          np.testing.assert_array_equal(loaded.solutions[name].diagnostics['nonlinear_iterations'],r.solutions[name].diagnostics['nonlinear_iterations'])
         plot_implicit_method_iterations({n:loaded.solutions[n] for n in ('BM4Implicit','GaussLegendre4')})
        plt.close("all")

    def test_recompute_dop853_and_reuse_radau(self):
        p=RandomPotentialConfig(amplitude=.2,max_wave_number=3,nx=16,ny=16,seed=27,interpolation_order=5).build()
        ic=latin_hypercube_gc_configuration(p,particle_count=3,seed=20260905,domain_margin_fraction=.35)
        c=FiveMethodComparisonConfig(t_span=(0.,.04),integration_step=.02,save_interval=.02,timing_warmups=0,timing_repeats=1,progress=False)
        original=run_five_method_comparison(p,ic,config=c)
        refined=run_five_method_comparison(
            p,
            ic,
            config=replace(c, integration_step=.005, reference_maximum_step=.01),
            method_names=('RK4',),
            reused_audit_reference=original.reference,
        )
        np.testing.assert_array_equal(refined.reference.audit_states, original.reference.audit_states)
        self.assertEqual(refined.reference.radau_runtime_seconds, original.reference.radau_runtime_seconds)
        self.assertEqual(refined.execution_log[0].method_name, 'DOP853')

    def test_parallel_model_campaigns(self):
        p=RandomPotentialConfig(amplitude=.2,max_wave_number=3,nx=16,ny=16,seed=27,interpolation_order=5).build()
        ic=latin_hypercube_gc_configuration(p,particle_count=3,seed=20260905,domain_margin_fraction=.35)
        c=FiveMethodComparisonConfig(t_span=(0.,.04),integration_step=.02,save_interval=.02,timing_warmups=0,timing_repeats=2,progress=False)
        reference=run_five_method_comparison(p,ic,config=replace(c, timing_repeats=1)).reference
        result=run_five_method_comparison(
            p,
            ic,
            config=c,
            method_names=('BM4Implicit','GaussLegendre4','RK4'),
            reused_reference=reference,
            parallel_models=True,
        )
        self.assertEqual(tuple(result.solutions), ('BM4Implicit','GaussLegendre4','RK4'))
        self.assertTrue(all(samples.size == 2 for samples in result.runtime_samples.values()))
        self.assertEqual(len([entry for entry in result.execution_log if entry.phase == 'timing']), 6)
