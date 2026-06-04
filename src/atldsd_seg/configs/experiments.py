"""Named experiment configurations for ATLDSD."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atldsd_seg.constants import CLASS_NAMES, NUM_CLASSES
from atldsd_seg.paths import DEFAULT_VOCDEVKIT_PATH, DEEPLABV3PLUS_ROOT, output_path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    backbone: str
    epochs: int
    input_size: tuple[int, int]
    batch_size: int
    freeze_batch_size: int
    init_lr: float
    dice_loss: bool = True
    focal_loss: bool = False
    attention_type: str = "none"
    use_ppm: bool = False
    seed: int = 11
    vocdevkit_path: Path = DEFAULT_VOCDEVKIT_PATH
    num_classes: int = NUM_CLASSES
    class_names: tuple[str, ...] = tuple(CLASS_NAMES)
    pretrained: bool = False
    model_path: Path | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def run_root(self) -> Path:
        return output_path("atldsd", self.name)

    @property
    def weights_dir(self) -> Path:
        return self.run_root / "weights"

    @property
    def logs_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def report_dir(self) -> Path:
        return self.run_root / "reports" / "best_val"


EXPERIMENTS = {
    "deeplabv3plus_mobilenet_150": ExperimentConfig(
        name="deeplabv3plus_mobilenet_150",
        model="deeplabv3plus",
        backbone="mobilenet",
        epochs=150,
        input_size=(256, 256),
        batch_size=4,
        freeze_batch_size=8,
        init_lr=0.0035,
        pretrained=False,
        model_path=DEEPLABV3PLUS_ROOT / "model_data" / "deeplab_mobilenetv2.pth",
    ),
    "deeplabv3plus_efficientnet_b4_150": ExperimentConfig(
        name="deeplabv3plus_efficientnet_b4_150",
        model="deeplabv3plus",
        backbone="efficientnet_b4",
        epochs=150,
        input_size=(256, 256),
        batch_size=2,
        freeze_batch_size=4,
        init_lr=0.001,
        pretrained=True,
        model_path=None,
    ),
    "deeplabv3plus_mobilenetv3_large_150": ExperimentConfig(
        name="deeplabv3plus_mobilenetv3_large_150",
        model="deeplabv3plus",
        backbone="mobilenetv3_large",
        epochs=150,
        input_size=(256, 256),
        batch_size=4,
        freeze_batch_size=8,
        init_lr=0.003,
        pretrained=True,
        model_path=None,
    ),
}


def get_experiment(name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}") from exc
