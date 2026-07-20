"""System entity and its GC/FC variants."""

from .fc import SystemFC
from .gc import SystemGC
from .observation import IntegrationStage, StageObserver

__all__ = ["IntegrationStage", "StageObserver", "SystemFC", "SystemGC"]
