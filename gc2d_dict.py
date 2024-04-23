###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/GC2D_intranet             ##
###################################################################################################

import numpy as xp

A = 0.7
rho = 0.1

Method = 'rotation_gc'
Ntraj = 50
Tf = 500

TwoStepIntegration = False
threshold = 4
Tmid = 200

TimeStep = 1e-1 
init = 'selected'
x0 = xp.linspace(0, 2 * xp.pi, Ntraj)
y0 = xp.pi * xp.ones(Ntraj)
ode_solver='BM4'

PlotResults = True
modulo = True
grid = False 
darkmode = True
fig_extension = '.pdf'
dpi = 200

SaveData = False
CheckEnergy = False
Parallelization = 1

M = 25

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
val_params = xp.meshgrid(A, rho, indexing='ij')
num_dict = len(val_params[0].flatten())

dict_list = [{'Method': Method} for _ in range(num_dict)]

if init == 'selected':
	for _, dict in enumerate(dict_list):
		dict.update({
			'x0': x0,
			'y0': y0})

for _, dict in enumerate(dict_list):
	dict.update({
		'A': val_params[0].flatten()[_],
		'rho': val_params[1].flatten()[_],
		'modulo': modulo,
		'threshold' : threshold,
		'grid': grid,
		'Ntraj': Ntraj,
		'Tf': Tf,
		'TwoStepIntegration' : TwoStepIntegration,
		'Tmid' : Tmid,
		'init': init,
		'TimeStep': TimeStep,
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'dpi': dpi,
		'darkmode': darkmode,
		'extension': fig_extension,
		'CheckEnergy': CheckEnergy,
		'M': M,
		'ode_solver': ode_solver})
###################################################################################################
