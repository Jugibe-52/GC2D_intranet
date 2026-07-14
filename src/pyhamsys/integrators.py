# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Coefficient definitions for explicit symplectic integrators."""

import numpy as xp


METHODS = ['Verlet', 'FR', 'Yo# with # any integer', 'Yos6', 'M2', 'M4', 'EFRL', 'PEFRL', 'VEFRL', 'BM4', 'BM6', 'RKN4b', 'RKN6b', 'RKN6a', 'ABA104', 'ABA864', 'ABA1064']


class SymplecticIntegrator:
	"""
    Some explicit symplectic splitting integrators in Python

    Attributes
    ----------
    name : str
        Name of the symplectic integrator.
		Integration methods are listed in https://pypi.org/project/pyhamsys/
    """
	def __repr__(self) -> str:
		return f'{self.__class__.__name__}({self.name})'

	def __str__(self) -> str:
		return f'{self.name}'

	def __init__(self, name:str) -> None:
		self.name = name
		if (self.name not in METHODS) and (self.name[:2] != 'Yo'):
			raise ValueError(f"The chosen integrator must be one of {METHODS}.")
		if self.name == 'Verlet':
			alpha_s = [0.5]
		elif self.name == 'FR':
			theta = 1 / (2 - 2**(1/3))
			alpha_s = [theta / 2, theta / 2, 0.5 - theta]
		elif self.name == 'Yos6':
			a = [0.784513610477560, 0.235573213359357, -1.17767998417887, 1.31518632068390]
			alpha_s = [a[0]/2,  a[0]/2, a[1]/2, a[1]/2, a[2]/2, a[2]/2, a[3]/2]
		elif self.name[:2] == 'Yo':
			try:
				alpha_s = xp.asarray([0.5])
				for n in range(1, int(self.name[2:]) // 2):
					x1 = 1 / (2 - 2**(1/(2*n+1)))
					x0 = 1 - 2 * x1
					alpha_ = xp.concatenate((alpha_s, xp.flip(alpha_s)))
					alpha_ = xp.concatenate((x1 * alpha_, x0 * alpha_s))
					alpha_s = alpha_.copy()
			except:
				raise NameError(f'{self.name} integrator not defined')
		elif self.name.endswith('EFRL'):
			if self.name.startswith('V'):
				xi, lam, chi = 0.1644986515575760, -0.02094333910398989, 1.235692651138917
			elif self.name.startswith('P'):
				xi, lam, chi = 0.1786178958448091, -0.2123418310626054, -0.06626458266981849
			else:
				xi, lam, chi = 0.1720865590295143, -0.09156203075515678, -0.1616217622107222
			alpha_s = [xi, 0.5 - lam - xi, lam + xi + chi -0.5, 0.5 - chi - xi]
		elif self.name == 'M2':
			y = (2*xp.sqrt(326)-36)**(1/3)
			z = (y**2 + 6 * y -2) / (12 * y)
			alpha_s = [z, 0.5 - z]
		elif self.name == 'M4':
			alpha_s = [(14-xp.sqrt(19))/108, (146+5*xp.sqrt(19))/540, (-23-20*xp.sqrt(19))/270, (-2+10*xp.sqrt(19))/135, 1/5]
		elif self.name == 'BM4':
			alpha_s = [0.0792036964311957, 0.1303114101821663, 0.2228614958676077, -0.3667132690474257, 0.3246481886897062, 0.1096884778767498]
		elif self.name == 'BM6':
			alpha_s = [0.050262764400392, 0.098553683500650, 0.314960616927694, -0.447346482695478, 0.492426372489876, -0.425118767797691, 0.237063913978122, 0.195602488600053, 0.346358189850727, -0.362762779254345]
		elif self.name == 'RKN4b':
			alpha_s = [0.082984406417405, 0.162314550766866, 0.233995250731502, 0.370877414979578, -0.409933719901926, 0.059762097006575]
		elif self.name == 'RKN6b':
			alpha_s = [0.041464998518262, 0.081764777428009, 0.116363894490058, 0.174189903309500, -0.214196095413653, 0.087146882788236, -0.011892898486655, -0.234438862575420, 0.222927475154732, 0.134281397641196, 0.102388527145735]
		elif self.name == 'RKN6a':
			alpha_s = [0.0378593198406116,  0.053859832783850, 0.048775800318585, 0.135207369686421, -0.161075257952980, 0.104540892120091, 0.209700510951356, -0.204785822176643, 0.074641362659228, 0.069119764509130, 0.037297935860413, 0.291269757886391, -0.300064001014902, 0.103652534528448]
		elif self.name == 'ABA104':
			alpha_s = [0.04706710064597251, 0.07181481672222451, 0.1129421186948636, 0.128108341856638, 0.1545976638231982, -0.4278843305285221, 0.4133542887856252]
		elif self.name == 'ABA864':
			alpha_s = [0.07113342649822312, 0.1119502609739741, 0.129203166982666, 0.1815796929159088, 0.3398320688569059, -0.3663966873688647, 0.03269807114118675]
		elif self.name == 'ABA1064':
			alpha_s = [0.03809449742241219, 0.05776438341466301, 0.08753433270225074, 0.116911820440748, 0.0907158752847932, 0.1263544726941979, 0.3095552309573282, -0.3269306129163933]
		elif self.name != 'Simple':
			raise NameError(f'{self.name} integrator not defined')
		if self.name == 'Simple':
			self.alpha_s = [1]
			self.alpha_o = [1]
		else:
			self.alpha_s = xp.concatenate((alpha_s, xp.flip(alpha_s)))
			self.alpha_o = xp.tile([1, 0], len(alpha_s))
