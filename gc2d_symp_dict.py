###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/GC2D_intranet             ##
###################################################################################################

A = 0.7

Ntraj = 5
Tf = 50
TimeStep = 1e-1  # recommended value: 5e-2 for interp, 1e-1 for symp
init = 'fixed'
solve_method = 'symp'
ode_solver = 'Verlet'
omega = 10

SaveData = True

M = 25
N = 2**10

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
dictparams = {
	'A': A,
	'Ntraj': Ntraj,
	'Tf': Tf,
	'init': init,
    'solve_method': solve_method,
    'ode_solver': ode_solver,
    'omega': omega,
	'TimeStep': TimeStep,
	'SaveData': SaveData,
	'M': M,
	'N': N}
###################################################################################################
