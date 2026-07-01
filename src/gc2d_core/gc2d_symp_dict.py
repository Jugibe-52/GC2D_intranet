"""Compatibility adapter for the legacy symplectic grid configuration."""

from .config import load_gc2dt_config

_config = load_gc2dt_config(config_group="assay", config_version="v_1", version="symplectic_grid")

dict_list = _config.cases()
Parallelization = _config.parallelization
