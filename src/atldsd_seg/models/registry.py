"""Model metadata used for reproducible paper experiments.

Implementation code for inherited baselines remains under ``src/models``.
This registry gives the paper code a stable, model-agnostic naming layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atldsd_seg.paths import DEEPLABV3PLUS_ROOT, SEGNEXT_ROOT


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    implementation_root: Path
    notes: str


MODEL_SPECS = {
    "deeplabv3plus": ModelSpec(
        name="deeplabv3plus",
        family="DeepLabV3+",
        implementation_root=DEEPLABV3PLUS_ROOT,
        notes="Main baseline and ablation carrier.",
    ),
    "segnext": ModelSpec(
        name="segnext",
        family="SegNeXt",
        implementation_root=SEGNEXT_ROOT,
        notes="Recent comparison model; requires mmseg-style config adaptation.",
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc
