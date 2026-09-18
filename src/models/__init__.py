"""
models/__init__.py
------------------
Exposes top-level models and adapters for clean imports.
"""

from .gcma_model import GCMAModel
from .hierarchical_gcma import HierarchicalGCMAModel
from .dssa_model import DSSAModel
from .dssa_directional_model import DSSADirectionalModel
from .directional_cross_attention import DirectionalCrossModalAttention
from .gated_residual_cross_attention import GatedResidualCrossModalAttention
from .spatial_adapter import SpatialDecompositionAdapter
from .pgmr_router import PhysicsGuidedRelevanceRouter
from .baseline_models import TabularOnlyModel, VisionOnlyModel, ConcatFusionModel

from .spatial_grid_gated_model import (
    SpatialGridGatedModel,
    ChemicalSpatialGatingBlock,
    DenseSpatialCrossAttentionBlock,
)

__all__ = [
    "GCMAModel",
    "HierarchicalGCMAModel",
    "DSSAModel",
    "DSSADirectionalModel",
    "SpatialGridGatedModel",
    "ChemicalSpatialGatingBlock",
    "DenseSpatialCrossAttentionBlock",
    "DirectionalCrossModalAttention",
    "GatedResidualCrossModalAttention",
    "SpatialDecompositionAdapter",
    "PhysicsGuidedRelevanceRouter",
    "TabularOnlyModel",
    "VisionOnlyModel",
    "ConcatFusionModel",
]
