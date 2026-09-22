"""Short matched-method checks for multi-region probe studies."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from diagnostics.poincare_probe import load_poincare_probe, save_poincare_probe
from potential import Grid, Potential
from simulation import BM4Midpoint, InitialValueProblem, SimulationRequest, simulate
from initial_conditions import GCInitialConfiguration
from dynamics import GuidingCenterDynamics
from studies.poincare_region_probe import (
    RegionProbeSettings, integrate_region_probes, region_probe_section, region_probe_panel,
)
from visualization.poincare_comparison import export_poincare_panel_comparison


class RegionProbeTests(unittest.TestCase):
    def setUp(self):
        self.settings = RegionProbeSettings('BM4Midpoint', (50,51), ((.25,.05),(.75,.95)),
            ('#a000d4','#111111'), 3, 20, 40, 0., coupling_frequency=np.pi/8)
        grid=Grid.periodic(32,16)
        self.potential=Potential(grid,mean=np.cos(grid.x)[:,None]*np.ones(grid.shape))

    def test_batched_midpoint_matches_public_global_clock_and_saves_its_diagnostic(self):
        events=[]
        actual=integrate_region_probes(self.potential,self.settings,progress=lambda a,b:events.append((a,b)))
        initial=np.asarray(self.settings.initial_xy_over_L)*self.potential.grid.period
        problem=InitialValueProblem(GuidingCenterDynamics(self.potential,rho=0.),
            GCInitialConfiguration.from_components(x=initial[:,0],y=initial[:,1]))
        expected=simulate(problem,BM4Midpoint(coupling_frequency=np.pi/8),
            SimulationRequest.uniform(t_span=(0,3),max_step=.05,sample_count=61))
        np.testing.assert_array_equal(actual['xy'],np.stack(expected.positions(),axis=-1).transpose(1,0,2))
        self.assertEqual(events,[(40,60),(60,60)])
        self.assertEqual(actual['method_diagnostics']['nonlinear_unknown_dimension'],0)
        self.assertEqual(actual['copy_separation_norms'].shape,(60,))
        with tempfile.TemporaryDirectory() as tmp:
            save_poincare_probe(Path(tmp),actual,contract={'method':'BM4Midpoint'})
            loaded,_=load_poincare_probe(Path(tmp))
            np.testing.assert_array_equal(loaded['copy_separation_norms'],actual['copy_separation_norms'])
        self.assertEqual(region_probe_section(actual,self.settings,self.potential.grid.period).shape,(4,2,2))

    def test_rk4_batch_recovers_opposite_constant_drifts_and_wraps_display(self):
        settings=replace(self.settings,method='RK4',coupling_frequency=None)
        result=integrate_region_probes(self.potential,settings)
        # With Phi=cos(x), x stays fixed and particles at pi/2 and 3*pi/2 drift oppositely.
        expected=np.asarray(settings.initial_xy_over_L)[:,1]*2*np.pi+result['times'][:,None]*[-1,1]
        np.testing.assert_allclose(result['xy'][:,:,1],expected,atol=3e-5)
        wrapped=region_probe_section(result,settings,2*np.pi)
        self.assertTrue(np.all((wrapped>=0)&(wrapped<1)))

    def test_regions_and_method_checks_prevent_mislabelled_panels(self):
        result=integrate_region_probes(self.potential,replace(self.settings,cycles=1))
        settings=replace(self.settings,cycles=1)
        background=SimpleNamespace(positions=np.full((2,1,2),.1),particle_ids=(1,),colors=('#123456',),
            metadata={'method':'BM4Midpoint','cycles':1,'field_provenance':{'grid':{'period':2*np.pi}}})
        panel=region_probe_panel(background,[(settings,result)])
        self.assertEqual(panel['particle_ids'],[1,50,51])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'viewer.html'
            export_poincare_panel_comparison(path,[panel],regions=[
                dict(label='Region 1',view=(.2,.1,.1),particle_id=50),
                dict(label='Region 2',view=(.6,.6,.2),particle_id=51)],highlight_particles=[50,51])
            self.assertIn('"highlightParticles": [50, 51]',path.read_text())
            with self.assertRaises(ValueError):
                export_poincare_panel_comparison(path,[panel],regions=[dict(label='Invalid',view=(.95,0,.1),particle_id=50)])
        with self.assertRaises(ValueError):
            region_probe_panel(background,[(settings,result),(settings,result)])
        with self.assertRaises(ValueError):
            replace(settings,method='RK4')


if __name__=='__main__':
    unittest.main()
