###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/GC2D_intranet             ##
###################################################################################################

import numpy as xp

A = 0.7
rho = xp.linspace(0.1, 0.3, 2)
eta = xp.linspace(-0.2, 0.3, 2)

traj_type = 'gc' # 'gc' (guiding centers) or 'fo' (full orbits) 
Ntraj = 5
Tf = 50
TimeStep = 1e-1  # recommended value: 5e-2 for interp, 1e-1 for symp
init = 'fixed'

SaveData = False
CheckEnergy = True
Parallelization = 2

M = 25

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
val_params = xp.meshgrid(A, rho, eta, indexing='ij')
num_dict = len(val_params[0].flatten())

dict_list = [{'traj_type': traj_type} for _ in range(num_dict)]

for _, dict in enumerate(dict_list):
	dict.update({
		'A': val_params[0].flatten()[_],
		'rho': val_params[1].flatten()[_],
		'eta': val_params[2].flatten()[_],
		'traj_type': traj_type,
		'Ntraj': Ntraj,
		'Tf': Tf,
		'init': init,
		'TimeStep': TimeStep,
		'SaveData': SaveData,
		'CheckEnergy': CheckEnergy,
		'M': M,
		'ode_solver': 'BM4'})
###################################################################################################
