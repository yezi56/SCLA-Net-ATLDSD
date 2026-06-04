"""Path helpers used by training, evaluation, and reporting code."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

DEFAULT_DATASET_ROOT = Path(r"D:\dataset\ATLDSD")
DEFAULT_VOCDEVKIT_PATH = DEFAULT_DATASET_ROOT / "VOCdevkit"

LEGACY_MODELS_ROOT = SRC_ROOT / "models"
DEEPLABV3PLUS_ROOT = LEGACY_MODELS_ROOT / "deeplabv3plus"
SEGNEXT_ROOT = LEGACY_MODELS_ROOT / "segnext"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def output_path(*parts: str) -> Path:
    return OUTPUTS_ROOT.joinpath(*parts)


def resolve_voc_root(vocdevkit_path: Path | str = DEFAULT_VOCDEVKIT_PATH, year: str = "VOC2007") -> Path:
    """Return the VOC year directory used by the inherited training code."""

    return Path(vocdevkit_path) / year
