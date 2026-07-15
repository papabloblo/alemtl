"""Model implementations for ALE-MTL."""

from .baselines import (
    CrossStitch,
    HardSharing,
    MMoE,
    MTAN,
    PLE,
    SingleTaskMLP,
    SoftSharing,
)
from .multitask_model import MultiTaskModel, SharedModule, SoftSharedModule

__all__ = [
    "CrossStitch",
    "HardSharing",
    "MMoE",
    "MTAN",
    "MultiTaskModel",
    "PLE",
    "SharedModule",
    "SingleTaskMLP",
    "SoftSharedModule",
    "SoftSharing",
]
