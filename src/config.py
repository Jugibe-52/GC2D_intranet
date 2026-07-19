"""Configuration loading for simulation entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np

from classes import Potential, PotentialHamsys, create_potential_hamsys
from contracts import (
	FourierParams,
	MockPotentialParams,
	OutputParams,
	ParameterMap,
	PotentialIntegrationParams,
	PotentialSourceKind,
	PotentialTrajectoryParams,
	PyHamSysParams,
	TrajectoryKind,
	TrajectoryParams,
)
from workflows.potentials import extract_potential, mock_potential

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CONFIG_SURFACE = "notebook"
DEFAULT_CONFIG_GROUP = "test"
DEFAULT_CONFIG_VERSION = "v_1"
DEFAULT_FOURIER_CONFIG = CONF_DIR / DEFAULT_CONFIG_SURFACE / "fourier" / DEFAULT_CONFIG_GROUP / f"{DEFAULT_CONFIG_VERSION}.json"
DEFAULT_POTENTIAL_CONFIG = CONF_DIR / DEFAULT_CONFIG_SURFACE / "potential" / DEFAULT_CONFIG_GROUP / f"{DEFAULT_CONFIG_VERSION}.json"
PYHAMSYS_PARAM_KEYS = {"TimeStep", "ode_solver", "CheckEnergy"}
OUTPUT_PARAM_KEYS = {"plot", "wrap"}


class ConfigError(ValueError):
	"""Raised when a configuration file is invalid."""


def config_path(
	name: str,
	config_surface: str = DEFAULT_CONFIG_SURFACE,
	config_group: str = DEFAULT_CONFIG_GROUP,
	config_version: str = DEFAULT_CONFIG_VERSION,
) -> Path:
	"""Return the default path for one config family."""
	kind = Path(name).stem
	if kind in {"fourier", "potential"}:
		new_path = CONF_DIR / config_surface / kind / config_group / f"{config_version}.json"
		if new_path.exists():
			return new_path
		legacy_family_path = CONF_DIR / kind / config_group / f"{config_version}.json"
		if legacy_family_path.exists():
			return legacy_family_path
		return CONF_DIR / config_group / config_version / f"{kind}.json"
	return CONF_DIR / config_group / config_version / name


def output_path_for_config(path: str | Path) -> Path:
	"""Return the output folder that mirrors one configuration path."""
	path = Path(path).resolve()
	try:
		relative = path.relative_to(CONF_DIR)
	except ValueError:
		return OUTPUTS_DIR / path.stem / "custom"
	parts = relative.parts
	if len(parts) < 3:
		return OUTPUTS_DIR / relative.with_suffix("")
	if parts[0] in {"notebook", "terminal"}:
		surface, kind, config_group, filename = parts[0], parts[1], parts[2], parts[-1]
		return OUTPUTS_DIR / surface / kind / config_group / Path(filename).stem
	if parts[0] in {"fourier", "potential"}:
		kind, config_group, filename = parts[0], parts[1], parts[-1]
		return OUTPUTS_DIR / kind / config_group / Path(filename).stem
	config_group, config_version, filename = parts[0], parts[1], parts[-1]
	return OUTPUTS_DIR / Path(filename).stem / config_group / config_version


def output_name_for_version(version: str) -> str:
	"""Return a compact filename prefix for one configuration profile."""
	return version.split("_", 1)[0]


def _read_json(path: str | Path) -> ParameterMap:
	path = Path(path)
	with path.open("r", encoding="utf-8") as file:
		data: object = json.load(file)
	if not isinstance(data, dict):
		raise ConfigError(f"Configuration root in {path} must be a JSON object.")
	return cast(ParameterMap, data)


def _read_config(path: str | Path) -> ParameterMap:
	path = Path(path)
	if path.suffix == ".json":
		return _read_json(path)
	raise ConfigError(f"Unsupported configuration format {path.suffix!r} in {path}.")


def _version_payload(data: ParameterMap, version: str | None, path: Path) -> tuple[str, ParameterMap]:
	if data.get("schema_version") != 1:
		raise ConfigError(f"{path} must declare schema_version=1.")
	versions = data.get("versions")
	if not isinstance(versions, dict) or not versions:
		raise ConfigError(f"{path} must define a non-empty 'versions' object.")
	selected = version or data.get("active_version")
	if not selected:
		raise ConfigError(f"{path} must define 'active_version' or receive a version argument.")
	if selected not in versions:
		available = ", ".join(sorted(versions))
		raise ConfigError(f"Version {selected!r} is not defined in {path}. Available: {available}.")
	payload = versions[selected]
	if not isinstance(payload, dict):
		raise ConfigError(f"Version {selected!r} in {path} must be a JSON object.")
	if not isinstance(selected, str):
		raise ConfigError(f"Selected version in {path} must be a string.")
	return selected, cast(ParameterMap, payload)


def _lookup_num(value: object, params: ParameterMap) -> int:
	if isinstance(value, str):
		if value not in params:
			raise ConfigError(f"Cannot resolve size reference {value!r}; key is missing.")
		value = params[value]
	if not isinstance(value, (int, float, np.integer)):
		raise ConfigError(f"Expected an integer size, got {value!r}.")
	return int(value)


def _as_int(value: object, name: str) -> int:
	if isinstance(value, (int, float, np.integer, np.floating, str)):
		return int(value)
	raise ConfigError(f"`{name}` must be an integer, got {value!r}.")


def _as_float(value: object, name: str) -> float:
	if isinstance(value, (int, float, np.integer, np.floating, str)):
		return float(value)
	raise ConfigError(f"`{name}` must be numeric, got {value!r}.")


def _optional_int(value: object, name: str) -> int | None:
	return None if value is None else _as_int(value, name)


def _expand_value(value: object, params: ParameterMap) -> object:
	if isinstance(value, dict):
		if "linspace" in value:
			args = value["linspace"]
			if not isinstance(args, list) or len(args) != 3:
				raise ConfigError("'linspace' values must be [start, stop, num].")
			return np.linspace(args[0], args[1], _lookup_num(args[2], params))
		if "constant" in value and "num" in value:
			return np.full(_lookup_num(value["num"], params), value["constant"])
		return {key: _expand_value(item, params) for key, item in value.items()}
	if isinstance(value, list):
		return [_expand_value(item, params) for item in value]
	return value


def _expand_params(params: ParameterMap) -> ParameterMap:
	expanded = params.copy()
	for key, value in list(expanded.items()):
		expanded[key] = _expand_value(value, expanded)
	return expanded


def _as_list(value: object) -> list[object]:
	if isinstance(value, list):
		return value
	return [value]


def _split_keys(params: ParameterMap, keys: set[str]) -> tuple[ParameterMap, ParameterMap]:
	remaining = params.copy()
	extracted = {}
	for key in keys:
		if key in remaining:
			extracted[key] = remaining.pop(key)
	return remaining, extracted


def _merge_config_block(
	base: ParameterMap,
	override: ParameterMap,
	block_name: str,
) -> tuple[ParameterMap, ParameterMap]:
	merged = base.copy()
	block = override.get(block_name, {})
	if not isinstance(block, dict):
		raise ConfigError(f"'{block_name}' overrides must be objects.")
	for key, value in override.items():
		if key != block_name:
			merged[key] = value
	return merged, block.copy()


def _normalize_symplectic_params(params: ParameterMap) -> FourierParams:
	from workflows.params import to_symp_params

	try:
		return to_symp_params(params)
	except (TypeError, ValueError) as exc:
		raise ConfigError(str(exc)) from exc


def _apply_fourier_output(defaults: ParameterMap, output: object, path: Path, selected: str) -> ParameterMap:
	"""Translate structured output options to the legacy FourierSystem flags."""
	if not isinstance(output, dict):
		raise ConfigError(f"'output' in {path}:{selected} must be an object.")
	params = defaults.copy()
	if "plot" in output:
		params["PlotResults"] = output["plot"]
		params["SavePlot"] = output["plot"]
	if "data" in output:
		params["SaveData"] = output["data"]
	for key in ("extension", "dpi"):
		if key in output:
			params[key] = output[key]
	return params


@dataclass(frozen=True)
class FourierConfig:
	"""Batch configuration for the Fourier/symplectic FourierSystem runner."""

	version: str
	defaults: ParameterMap
	pyhamsys: ParameterMap = field(default_factory=dict)
	sweep: dict[str, list[object]] = field(default_factory=dict)
	case_overrides: list[ParameterMap] = field(default_factory=list)
	parallelization: int | str = 1
	output_dir: Path | None = None
	output_name: str | None = None

	def cases(self) -> list[FourierParams]:
		if self.case_overrides:
			raw_cases = []
			for override in self.case_overrides:
				case, case_pyhamsys = _merge_config_block(self.defaults, override, "pyhamsys")
				case, legacy_pyhamsys = _split_keys(case, PYHAMSYS_PARAM_KEYS)
				raw_cases.append(case | self.pyhamsys | legacy_pyhamsys | case_pyhamsys)
		elif self.sweep:
			keys = list(self.sweep)
			raw_cases = []
			for values in product(*[_as_list(self.sweep[key]) for key in keys]):
				case = self.defaults.copy()
				case.update(dict(zip(keys, values, strict=True)))
				case, legacy_pyhamsys = _split_keys(case, PYHAMSYS_PARAM_KEYS)
				raw_cases.append(case | self.pyhamsys | legacy_pyhamsys)
		else:
			case, legacy_pyhamsys = _split_keys(self.defaults, PYHAMSYS_PARAM_KEYS)
			raw_cases = [case | self.pyhamsys | legacy_pyhamsys]
		normalized_cases = [_normalize_symplectic_params(_expand_params(case)) for case in raw_cases]
		if self.output_dir is not None:
			for normalized_case in normalized_cases:
				normalized_case["output_dir"] = str(self.output_dir)
		if self.output_name is not None:
			for normalized_case in normalized_cases:
				normalized_case["output_name"] = self.output_name
		return normalized_cases


@dataclass(frozen=True)
class PotentialConfig:
	type: PotentialSourceKind
	path: Path | None = None
	B: float = 1
	indx: list[int] | None = None
	nx: int | None = None
	ny: int | None = None
	denoising: bool = False
	sigma: float = 1
	k: int = 3
	mock: MockPotentialParams = field(default_factory=lambda: cast(MockPotentialParams, {}))

	def build(self) -> Potential:
		if self.type not in {"hdf5", "mock", "hdf5_or_mock"}:
			raise ConfigError(f"Invalid potential type {self.type!r}.")
		if self.type in {"hdf5", "hdf5_or_mock"} and self.path is not None and self.path.exists():
			target_shape = None if self.nx is None and self.ny is None else (self.nx, self.ny)
			return extract_potential(
				self.path,
				B=self.B,
				indx=self.indx,
				target_shape=target_shape,
				denoising=self.denoising,
				sigma=self.sigma,
				k=self.k,
			)
		if self.type == "hdf5":
			raise ConfigError(f"Potential file {self.path} does not exist.")
		mock = self.mock
		return mock_potential(
			mock.get("A", 1 / self.B),
			mock.get("M", 25),
			self.nx or mock.get("nx", 128),
			self.ny or mock.get("ny", 128),
			seed=mock.get("seed", 27),
			k=self.k,
		)


@dataclass(frozen=True)
class PotentialRunConfig:
	"""Configuration for the HDF5/mock potential runner."""

	version: str
	potential: PotentialConfig
	trajectory: PotentialTrajectoryParams
	integration: PotentialIntegrationParams
	pyhamsys: PyHamSysParams = field(default_factory=lambda: cast(PyHamSysParams, {}))
	output: OutputParams = field(default_factory=lambda: cast(OutputParams, {}))
	output_dir: Path | None = None
	output_name: str | None = None

	def build_system(self) -> PotentialHamsys:
		potential = self.potential.build()
		traj: TrajectoryParams = {
			"type": self.trajectory.get("type", "gc"),
			"rho": self.trajectory.get("rho", 0),
			"eta": self.trajectory.get("eta", 0),
		}
		return create_potential_hamsys(potential, traj)

	def initial_condition_count(self) -> int:
		return int(self.trajectory.get("Ntraj", 20))

	def initial_condition_type(self) -> Literal["random", "fixed"]:
		return self.trajectory.get("init", "fixed")


def load_fourier_config(
	path: str | Path | None = None,
	version: str | None = None,
	config_surface: str = DEFAULT_CONFIG_SURFACE,
	config_group: str = DEFAULT_CONFIG_GROUP,
	config_version: str = DEFAULT_CONFIG_VERSION,
) -> FourierConfig:
	path = Path(path) if path is not None else config_path("fourier", config_surface, config_group, config_version)
	selected, payload = _version_payload(_read_config(path), version, path)
	defaults = payload.get("defaults", {})
	if not isinstance(defaults, dict):
		raise ConfigError(f"'defaults' in {path}:{selected} must be an object.")
	defaults = _apply_fourier_output(cast(ParameterMap, defaults), payload.get("output", {}), path, selected)
	defaults, legacy_pyhamsys = _split_keys(defaults, PYHAMSYS_PARAM_KEYS)
	pyhamsys = payload.get("pyhamsys", {})
	if not isinstance(pyhamsys, dict):
		raise ConfigError(f"'pyhamsys' in {path}:{selected} must be an object.")
	sweep = payload.get("sweep", {})
	if not isinstance(sweep, dict):
		raise ConfigError(f"'sweep' in {path}:{selected} must be an object.")
	cases = payload.get("cases", [])
	if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
		raise ConfigError(f"'cases' in {path}:{selected} must be a list of objects.")
	parallelization = payload.get("parallelization", 1)
	if not isinstance(parallelization, (int, str)):
		raise ConfigError("`parallelization` must be an integer or 'all'.")
	output_name = payload.get("output_name", output_name_for_version(selected))
	if output_name is not None and not isinstance(output_name, str):
		raise ConfigError("`output_name` must be a string.")
	return FourierConfig(
		version=selected,
		defaults=defaults,
		pyhamsys=legacy_pyhamsys | cast(ParameterMap, pyhamsys),
		sweep={key: _as_list(value) for key, value in sweep.items()},
		case_overrides=cast(list[ParameterMap], cases),
		parallelization=parallelization,
		output_dir=output_path_for_config(path),
		output_name=output_name,
	)


def load_potential_config(
	path: str | Path | None = None,
	version: str | None = None,
	config_surface: str = DEFAULT_CONFIG_SURFACE,
	config_group: str = DEFAULT_CONFIG_GROUP,
	config_version: str = DEFAULT_CONFIG_VERSION,
) -> PotentialRunConfig:
	path = Path(path) if path is not None else config_path("potential", config_surface, config_group, config_version)
	selected, payload = _version_payload(_read_config(path), version, path)
	potential_payload = payload.get("potential", {})
	if not isinstance(potential_payload, dict):
		raise ConfigError(f"'potential' in {path}:{selected} must be an object.")
	potential_path = potential_payload.get("path")
	potential_type = potential_payload.get("type", "mock")
	if potential_type not in {"hdf5", "mock", "hdf5_or_mock"}:
		raise ConfigError(f"Invalid potential type {potential_type!r}.")
	raw_mock = potential_payload.get("mock", {})
	if not isinstance(raw_mock, dict):
		raise ConfigError("`potential.mock` must be an object.")
	mock: MockPotentialParams = {
		"A": _as_float(raw_mock.get("A", 1 / _as_float(potential_payload.get("B", 1), "potential.B")), "potential.mock.A"),
		"M": _as_int(raw_mock.get("M", 25), "potential.mock.M"),
		"nx": _as_int(raw_mock.get("nx", 128), "potential.mock.nx"),
		"ny": _as_int(raw_mock.get("ny", 128), "potential.mock.ny"),
		"seed": _as_int(raw_mock.get("seed", 27), "potential.mock.seed"),
	}
	raw_indices = potential_payload.get("indx")
	if raw_indices is not None and not isinstance(raw_indices, list):
		raise ConfigError("`potential.indx` must be a list of integers.")
	potential = PotentialConfig(
		type=cast(PotentialSourceKind, potential_type),
		path=Path(potential_path).expanduser() if potential_path else None,
		B=_as_float(potential_payload.get("B", 1), "potential.B"),
		indx=None if raw_indices is None else [_as_int(item, "potential.indx") for item in raw_indices],
		nx=_optional_int(potential_payload.get("nx"), "potential.nx"),
		ny=_optional_int(potential_payload.get("ny"), "potential.ny"),
		k=_as_int(potential_payload.get("k", 3), "potential.k"),
		denoising=bool(potential_payload.get("denoising", False)),
		sigma=_as_float(potential_payload.get("sigma", 1), "potential.sigma"),
		mock=mock,
	)
	trajectory = payload.get("trajectory", {})
	integration = payload.get("integration", {})
	pyhamsys = payload.get("pyhamsys", {})
	output = payload.get("output", {})
	if not all(isinstance(block, dict) for block in (trajectory, integration, pyhamsys, output)):
		raise ConfigError(f"'trajectory', 'integration', 'pyhamsys' and 'output' in {path}:{selected} must be objects.")
	trajectory = cast(ParameterMap, trajectory)
	integration = cast(ParameterMap, integration)
	pyhamsys = cast(ParameterMap, pyhamsys)
	output = cast(ParameterMap, output)
	integration, legacy_pyhamsys = _split_keys(integration, PYHAMSYS_PARAM_KEYS)
	integration, legacy_output = _split_keys(integration, OUTPUT_PARAM_KEYS)
	trajectory_type = trajectory.get("type", "gc")
	if trajectory_type not in {"gc", "fo"}:
		raise ConfigError(f"Invalid trajectory type {trajectory_type!r}.")
	init_value = trajectory.get("init", "fixed")
	if init_value not in {"random", "fixed"}:
		raise ConfigError(f"Invalid potential initial-condition type {init_value!r}.")
	typed_trajectory: PotentialTrajectoryParams = {
		"type": cast(TrajectoryKind, trajectory_type),
		"rho": _as_float(trajectory.get("rho", 0), "trajectory.rho"),
		"eta": _as_float(trajectory.get("eta", 0), "trajectory.eta"),
		"Ntraj": _as_int(trajectory.get("Ntraj", 20), "trajectory.Ntraj"),
		"init": cast(Literal['random', 'fixed'], init_value),
	}
	output_name = payload.get("output_name", output_name_for_version(selected))
	if output_name is not None and not isinstance(output_name, str):
		raise ConfigError("`output_name` must be a string.")
	return PotentialRunConfig(
		version=selected,
		potential=potential,
		trajectory=typed_trajectory,
		integration=cast(PotentialIntegrationParams, integration),
		pyhamsys=cast(PyHamSysParams, legacy_pyhamsys | pyhamsys),
		output=cast(OutputParams, legacy_output | output),
		output_dir=output_path_for_config(path),
		output_name=output_name,
	)
