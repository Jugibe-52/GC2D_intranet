###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/GC2D_intranet             ##
###################################################################################################

A = 0.7

Ntraj = 5
Tf = 50
TimeStep = 1e-1  # recommended value: 5e-2 for interp, 1e-1 for symp
init = 'fixed'
ode_solver = 'BM4'

SaveData = False
CheckEnergy = True

M = 25

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
dictparams = {
	'A': A,
	'Ntraj': Ntraj,
	'Tf': Tf,
	'init': init,
    'ode_solver': ode_solver,
	'TimeStep': TimeStep,
	'SaveData': SaveData,
    'CheckEnergy': CheckEnergy,
	'M': M}
###################################################################################################
