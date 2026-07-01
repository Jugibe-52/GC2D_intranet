"""Configuration loading for simulation entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

from classes import PotentialSystem, Potential
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


def _read_json(path: str | Path) -> dict[str, Any]:
	path = Path(path)
	with path.open("r", encoding="utf-8") as file:
		data = json.load(file)
	if not isinstance(data, dict):
		raise ConfigError(f"Configuration root in {path} must be a JSON object.")
	return data


def _read_config(path: str | Path) -> dict[str, Any]:
	path = Path(path)
	if path.suffix == ".json":
		return _read_json(path)
	raise ConfigError(f"Unsupported configuration format {path.suffix!r} in {path}.")


def _version_payload(data: dict[str, Any], version: str | None, path: Path) -> tuple[str, dict[str, Any]]:
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
	return selected, payload


def _lookup_num(value: Any, params: dict[str, Any]) -> int:
	if isinstance(value, str):
		if value not in params:
			raise ConfigError(f"Cannot resolve size reference {value!r}; key is missing.")
		value = params[value]
	return int(value)


def _expand_value(value: Any, params: dict[str, Any]) -> Any:
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


def _expand_params(params: dict[str, Any]) -> dict[str, Any]:
	expanded = params.copy()
	for key, value in list(expanded.items()):
		expanded[key] = _expand_value(value, expanded)
	return expanded


def _as_list(value: Any) -> list[Any]:
	if isinstance(value, list):
		return value
	return [value]


def _split_keys(params: dict[str, Any], keys: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
	remaining = params.copy()
	extracted = {}
	for key in keys:
		if key in remaining:
			extracted[key] = remaining.pop(key)
	return remaining, extracted


def _merge_config_block(
	base: dict[str, Any],
	override: dict[str, Any],
	block_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
	merged = base.copy()
	block = override.get(block_name, {})
	if block and not isinstance(block, dict):
		raise ConfigError(f"'{block_name}' overrides must be objects.")
	for key, value in override.items():
		if key != block_name:
			merged[key] = value
	return merged, block.copy()


def _normalize_symplectic_params(params: dict[str, Any]) -> dict[str, Any]:
	params = params.copy()
	method = params.get("Method")
	if "traj_type" not in params:
		params["traj_type"] = method.rsplit("_", 1)[-1] if isinstance(method, str) else "gc"
	if params["traj_type"] not in {"gc", "fo"}:
		raise ConfigError(f"Invalid traj_type={params['traj_type']!r}; expected 'gc' or 'fo'.")
	params.setdefault("eta", params.get("rho", 0))
	if params["traj_type"] == "fo" and params["eta"] == 0:
		raise ConfigError("Full-orbit integrations require a non-zero 'eta' parameter.")
	return params


def _apply_fourier_output(defaults: dict[str, Any], output: dict[str, Any], path: Path, selected: str) -> dict[str, Any]:
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
	defaults: dict[str, Any]
	pyhamsys: dict[str, Any] = field(default_factory=dict)
	sweep: dict[str, list[Any]] = field(default_factory=dict)
	case_overrides: list[dict[str, Any]] = field(default_factory=list)
	parallelization: int | str = 1
	output_dir: Path | None = None
	output_name: str | None = None

	def cases(self) -> list[dict[str, Any]]:
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
		cases = [_normalize_symplectic_params(_expand_params(case)) for case in raw_cases]
		if self.output_dir is not None:
			for case in cases:
				case["output_dir"] = str(self.output_dir)
		if self.output_name is not None:
			for case in cases:
				case["output_name"] = self.output_name
		return cases


@dataclass(frozen=True)
class PotentialConfig:
	type: str
	path: Path | None = None
	B: float = 1
	indx: list[int] | None = None
	nx: int | None = None
	ny: int | None = None
	denoising: bool = False
	sigma: float = 1
	mock: dict[str, Any] = field(default_factory=dict)

	def build(self) -> Potential:
		if self.type not in {"hdf5", "mock", "hdf5_or_mock"}:
			raise ConfigError(f"Invalid potential type {self.type!r}.")
		if self.type in {"hdf5", "hdf5_or_mock"} and self.path is not None and self.path.exists():
			return extract_potential(
				self.path,
				B=self.B,
				indx=self.indx,
				nx=self.nx,
				ny=self.ny,
				denoising=self.denoising,
				sigma=self.sigma,
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
		)


@dataclass(frozen=True)
class PotentialRunConfig:
	"""Configuration for the HDF5/mock potential runner."""

	version: str
	potential: PotentialConfig
	trajectory: dict[str, Any]
	integration: dict[str, Any]
	pyhamsys: dict[str, Any] = field(default_factory=dict)
	output: dict[str, Any] = field(default_factory=dict)
	output_dir: Path | None = None
	output_name: str | None = None

	def build_system(self) -> PotentialSystem:
		potential = self.potential.build()
		traj = {
			"type": self.trajectory.get("type", "gc"),
			"rho": self.trajectory.get("rho", 0),
			"eta": self.trajectory.get("eta", 0),
		}
		return PotentialSystem(potential, traj, k=self.trajectory.get("k", 3))

	def initial_condition_count(self) -> int:
		return int(self.trajectory.get("Ntraj", 20))

	def initial_condition_type(self) -> str:
		return str(self.trajectory.get("init", "fixed"))


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
	defaults = _apply_fourier_output(defaults, payload.get("output", {}), path, selected)
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
	return FourierConfig(
		version=selected,
		defaults=defaults,
		pyhamsys=legacy_pyhamsys | pyhamsys,
		sweep={key: _as_list(value) for key, value in sweep.items()},
		case_overrides=cases,
		parallelization=payload.get("parallelization", 1),
		output_dir=output_path_for_config(path),
		output_name=payload.get("output_name", output_name_for_version(selected)),
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
	potential = PotentialConfig(
		type=potential_payload.get("type", "mock"),
		path=Path(potential_path).expanduser() if potential_path else None,
		B=potential_payload.get("B", 1),
		indx=potential_payload.get("indx"),
		nx=potential_payload.get("nx"),
		ny=potential_payload.get("ny"),
		denoising=potential_payload.get("denoising", False),
		sigma=potential_payload.get("sigma", 1),
		mock=potential_payload.get("mock", {}),
	)
	trajectory = payload.get("trajectory", {})
	integration = payload.get("integration", {})
	pyhamsys = payload.get("pyhamsys", {})
	output = payload.get("output", {})
	if not all(isinstance(block, dict) for block in (trajectory, integration, pyhamsys, output)):
		raise ConfigError(f"'trajectory', 'integration', 'pyhamsys' and 'output' in {path}:{selected} must be objects.")
	integration, legacy_pyhamsys = _split_keys(integration, PYHAMSYS_PARAM_KEYS)
	integration, legacy_output = _split_keys(integration, OUTPUT_PARAM_KEYS)
	return PotentialRunConfig(
		version=selected,
		potential=potential,
		trajectory=trajectory,
		integration=integration,
		pyhamsys=legacy_pyhamsys | pyhamsys,
		output=legacy_output | output,
		output_dir=output_path_for_config(path),
		output_name=payload.get("output_name", output_name_for_version(selected)),
	)
