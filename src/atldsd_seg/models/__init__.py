"""Model registry for experiments."""

from .clcs_deeplabv3plus import CLCSDeepLabV3Plus
from .registry import MODEL_SPECS, ModelSpec, get_model_spec

__all__ = ["CLCSDeepLabV3Plus", "MODEL_SPECS", "ModelSpec", "get_model_spec"]
