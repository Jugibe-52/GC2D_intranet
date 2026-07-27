"""Compatibility system facades and legacy result/observer exports."""

from .fc import SystemFC
from .gc import SystemGC
from .observation import IntegrationStage, StageObserver
from .solution import Solution

__all__ = ["IntegrationStage", "Solution", "StageObserver", "SystemFC", "SystemGC"]
