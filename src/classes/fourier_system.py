import logging

import numpy as xp
from pyhamsys import HamSys
from scipy.special import jv

from contracts import FourierParams, InitialConditionKind, TrajectoryKind

logger = logging.getLogger(__name__)


def real_imag(z: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
	return z.real, z.imag


class FourierSystem(HamSys):
	traj_type: TrajectoryKind
	M: int
	A: float
	rho: float
	eta: float
	Ntraj: int
	Tf: int
	TimeStep: float
	ode_solver: str
	CheckEnergy: bool
	init: InitialConditionKind

	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	def __init__(self, dict_: FourierParams) -> None:
		super().__init__(ndof=1.5 if dict_['traj_type']=='gc' else 2.5)
		self.DictParams: FourierParams = dict_.copy()
		self.traj_type = dict_['traj_type']
		self.M = dict_['M']
		self.A = dict_['A']
		self.rho = dict_['rho']
		self.eta = dict_['eta']
		self.Ntraj = dict_['Ntraj']
		self.Tf = dict_['Tf']
		self.TimeStep = dict_['TimeStep']
		self.ode_solver = dict_['ode_solver']
		self.CheckEnergy = dict_['CheckEnergy']
		self.init = dict_['init']
		self.Method = dict_.get('Method', f'poincare_{self.traj_type}')
		self.TwoStepIntegration = dict_.get('TwoStepIntegration', False)
		self.Tmid = dict_.get('Tmid', 0)
		self.threshold = dict_.get('threshold', 4.0)
		self.thresh_b = dict_.get('thresh_b', 1.5)
		self.x0 = xp.asarray(dict_.get('x0', xp.empty(0)))
		self.y0 = xp.asarray(dict_.get('y0', xp.empty(0)))
		self.modulo = dict_.get('modulo', False)
		self.grid = dict_.get('grid', False)
		self.darkmode = dict_.get('darkmode', False)
		self.PlotResults = dict_.get('PlotResults', False)
		self.SavePlot = dict_.get('SavePlot', False)
		self.SaveData = dict_.get('SaveData', False)
		self.extension = dict_.get('extension', '.png')
		self.dpi = dict_.get('dpi', 200)
		self.output_dir = dict_.get('output_dir', '.')
		self.output_name = dict_.get('output_name', 'notebook')
		logger.info(
			"Initializing FourierSystem: traj=%s M=%s A=%s rho=%s eta=%s Ntraj=%s Tf=%s",
			self.traj_type,
			self.M,
			self.A,
			self.rho,
			getattr(self, 'eta', 'n/a'),
			self.Ntraj,
			self.Tf,
		)
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5 * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		if self.traj_type == 'gc':
			flr1_coeff = jv(0, self.rho * sqrt_nm)
			self.phic *= flr1_coeff
			logger.debug("Applied FLR correction for guiding-center trajectory: rho=%s", self.rho)

		self.fft_phi_ = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	
		active_modes = int(xp.count_nonzero(self.phic))
		logger.info("FourierSystem initialized with %d active Fourier modes", active_modes)

	def initial_conditions(self, type: InitialConditionKind = 'fixed') -> xp.ndarray:
		original_ntraj = self.Ntraj
		logger.info("Generating initial conditions: type=%s traj=%s Ntraj=%s", type, self.traj_type, self.Ntraj)
		if type == 'random':
			y0 = 2 * xp.pi * xp.random.rand(2 * self.Ntraj)
		elif type == 'fixed':
			self.Ntraj = int(xp.sqrt(self.Ntraj))**2
			if self.Ntraj != original_ntraj:
				logger.info("Adjusted Ntraj from %s to square grid size %s for fixed initialization", original_ntraj, self.Ntraj)
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(self.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
		elif type == 'selected':
			x0 = xp.asarray(self.x0)
			y0_selected = xp.asarray(self.y0)
			if x0.shape != y0_selected.shape:
				raise ValueError("`x0` and `y0` must have the same shape when init='selected'.")
			self.Ntraj = x0.size
			logger.info("Using selected initial conditions with %d trajectories", self.Ntraj)
			y0 = xp.concatenate((x0.ravel(), y0_selected.ravel()), axis=None)
		else:
			raise ValueError("`type` must be 'random', 'fixed' or 'selected'.")
		if self.traj_type == 'fo':
			phi_perp = 2 * xp.pi * xp.random.rand(self.Ntraj)
			y0 = xp.concatenate((y0, xp.cos(phi_perp), xp.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				y0 = xp.concatenate((y0, xp.zeros(self.Ntraj)), axis=None)
			logger.debug("Added full-orbit velocity variables: CheckEnergy=%s", self.CheckEnergy)
		logger.info("Initial condition vector ready: shape=%s", y0.shape)
		return y0

	def y_dot(self, t: float, y: xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.asarray(xp.einsum('ijk,jk...->i...', self.fft_phi_, exp_xy).real).reshape(y.shape)
	
	def k_dot(self, t: float, y: xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.asarray(xp.einsum('jk,jk...->...', self.phic, exp_xy).real)
	
	def potential(self, t: float | xp.ndarray, y: xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.asarray(xp.einsum('jk,jk...->...', self.phic, exp_xy).imag)
	
	def hamiltonian(self, t: float | xp.ndarray, y: xp.ndarray) -> xp.ndarray:
		if self.traj_type == 'gc':
			return self.potential(t, y)
		x_, y_, vx, vy = xp.split(y, 4)
		return xp.asarray(self.rho / (4 * xp.abs(self.eta)) * (vx**2 + vy**2) + self.potential(t, xp.concatenate((x_, y_), axis=0)) * xp.sign(self.eta) / self.rho)
	
	def chi(self, h: float, t: float, y: xp.ndarray) -> xp.ndarray:
		if self.CheckEnergy:
			x_, y_, vx, vy, k = xp.split(y, 5)
		else:
			x_, y_, vx, vy = xp.split(y, 4)
		exp_ = xp.exp(-1j * h / (2 * self.eta))
		x_, y_ = real_imag(x_ + 1j * y_ + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = xp.split(self.y_dot(t, xp.concatenate((x_, y_), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * xp.sign(self.eta) / self.rho)
		if not self.CheckEnergy:
			return xp.concatenate((x_, y_, vx, vy), axis=None)
		k += h * xp.sign(self.eta) / self.rho * self.k_dot(t, xp.concatenate((x_, y_), axis=None)) 
		return xp.concatenate((x_, y_, vx, vy, k), axis=None)
	
	def chi_star(self, h: float, t: float, y: xp.ndarray) -> xp.ndarray:
		if self.CheckEnergy:
			x_, y_, vx, vy, k = xp.split(y, 5)
		else:
			x_, y_, vx, vy = xp.split(y, 4)
		pot = xp.split(self.y_dot(t, xp.concatenate((x_, y_), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * xp.sign(self.eta) / self.rho)
		if self.CheckEnergy:
			k += h * xp.sign(self.eta) / self.rho * self.k_dot(t, xp.concatenate((x_, y_), axis=None))
		exp_ = xp.exp(-1j * h / (2 * self.eta))
		x_, y_ = real_imag(x_ + 1j * y_ + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		if not self.CheckEnergy:
			return xp.concatenate((x_, y_, vx, vy), axis=None)
		return xp.concatenate((x_, y_, vx, vy, k), axis=None)
