"""Single-run energy-balance and trajectory study with a DOP853 reference."""
import numpy as np
from scipy.integrate import solve_ivp
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from simulation import InitialValueProblem, SimulationRequest, BM4Implicit, GaussLegendre4, RK4, simulate
from diagnostics.bm4_energy_balance import BM4EnergyBalanceObserver

METHODS = ("DOP853", "BM4Implicit", "GaussLegendre4", "RK4")


def horizontal_configuration(potential, x_fractions, y_fraction):
    """Place three equidistant particles on one horizontal periodic-cell line."""
    x = np.asarray(x_fractions, dtype=float)
    if x.shape != (3,) or not np.all((x > 0) & (x < 1)) or not 0 < y_fraction < 1:
        raise ValueError("Use three interior x fractions and one interior y fraction.")
    if not np.all(np.diff(x) > 0) or not np.isclose(*np.diff(x)):
        raise ValueError("The three x fractions must increase with equal spacing.")
    return GCInitialConfiguration.from_components(
        x=potential.grid.xmin + potential.grid.period*x,
        y=np.full(3, potential.grid.ymin + potential.grid.period*y_fraction))


def run_energy_balance(potential, configuration, config):
    """Integrate each method once; advance energy at stages, then save sparsely.

    DOP853 controls the augmented (x,y,kappa) system adaptively with max_step=h.
    Fixed methods take exactly h. All persisted arrays use the same output grid.
    """
    cycles = config['cycles']
    steps = config['steps_per_cycle']
    saves = config['saves_per_cycle']
    if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in (cycles, steps, saves)) or steps % saves:
        raise ValueError("Positive integer cycles and step counts with steps divisible by saves are required.")
    h = config['cycle_duration']/steps
    duration = cycles*config['cycle_duration']
    times = np.linspace(0, duration, int(cycles*saves)+1)
    dynamics = GuidingCenterDynamics(potential, rho=config['rho'])
    problem = InitialValueProblem(dynamics, configuration)
    z0 = problem.initial_state
    n = z0.size//2
    if n != 3:
        raise ValueError("This study requires three particles.")
    request = SimulationRequest((0., duration), h, times)
    arrays = {'times': times, 'initial_state': z0,
              'domain': np.array([potential.grid.xmin, potential.grid.ymin, potential.grid.period])}
    def rhs(t, value):
        return np.concatenate((dynamics.vector_field(t, value[:2*n]),
                               dynamics.extended_momentum_derivative(t, value[:2*n])))
    print('Starting DOP853', flush=True)
    reference = solve_ivp(rhs, request.t_span, np.r_[z0, np.zeros(n)], method='DOP853',
                          t_eval=times, max_step=h, rtol=config['reference_rtol'], atol=config['reference_atol'])
    if not reference.success:
        raise RuntimeError(reference.message)
    arrays['DOP853.states'] = reference.y[:2*n]
    arrays['DOP853.kappa'] = reference.y[2*n:]
    common = dict(newton_absolute_tolerance=config['newton_atol'],
                  newton_relative_tolerance=config['newton_rtol'],
                  newton_max_iterations=config['newton_max_iterations'],
                  newton_jacobian_method='analytic',
                  newton_jacobian_relative_step=config['jacobian_relative_step'])
    observer = BM4EnergyBalanceObserver(dynamics, n, steps//saves)
    methods = {
        'BM4Implicit': BM4Implicit(**common, coupling_frequency=config['coupling_frequency'], step_observer=observer),
        'GaussLegendre4': GaussLegendre4(**common, track_energy=True),
        'RK4': RK4(track_energy=True),
    }
    for name, method in methods.items():
        print(f'Starting {name}', flush=True)
        solution = simulate(problem, method, request)
        assert solution.diagnostics['step_count'] == cycles*steps
        arrays[name+'.states'] = solution.states
        if name == 'BM4Implicit':
            assert not observer.block and observer.step_count == cycles*steps
            arrays[name+'.kappa'] = np.asarray(observer.momenta).T
            arrays['BM4Implicit.mu_global_statistics'] = np.asarray(observer.global_statistics).T
            for kind in ('endpoint', 'max', 'rms', 'mean'):
                arrays['BM4Implicit.mu_'+kind] = np.asarray(getattr(observer, 'mu_'+kind)).T
        else:
            arrays[name+'.kappa'] = solution.diagnostics['extended_momentum']
    for name in METHODS:
        states = arrays[name+'.states']
        energy = dynamics.hamiltonian(times, states)
        arrays[name+'.H'] = energy
        arrays[name+'.balance'] = energy + arrays[name+'.kappa'] - energy[:, :1]
        delta = states - arrays['DOP853.states']
        delta = (delta + potential.grid.period/2) % potential.grid.period - potential.grid.period/2
        arrays[name+'.distance'] = np.linalg.norm(delta.reshape(2, n, -1), axis=0)
    if any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError('Nonfinite study results.')
    print('All four integrations completed; no timing campaign was run.', flush=True)
    return arrays
