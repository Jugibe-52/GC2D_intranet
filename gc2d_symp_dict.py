###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/GC2D_intranet             ##
###################################################################################################

import numpy as xp

A = 0.7
rho = xp.linspace(0.1, 0.3, 2)
eta = 0.001

traj_type = 'fo' # 'gc' (guiding centers) or 'fo' (full orbits) 
Ntraj = 5
Tf = 50
TimeStep = 1e-3  # recommended value: 5e-2 for interp, 1e-1 for symp
init = 'fixed'

SaveData = False
CheckEnergy = True
Parallelization = 1

M = 25

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################

dict_list = [ [] for _ in range(len(rho)) ]

for _, rho_ in enumerate(rho):
	dict_list[_] = {
		'A': A,
		'rho': rho_,
		'eta': eta,
		'traj_type': traj_type,
		'Ntraj': Ntraj,
		'Tf': Tf,
		'init': init,
		'TimeStep': TimeStep,
		'SaveData': SaveData,
		'CheckEnergy': CheckEnergy,
		'M': M,
		'ode_solver': 'BM4'}
###################################################################################################
