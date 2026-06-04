"""Path helpers used by training, evaluation, and reporting code."""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def _first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return fallback


PROJECT_ROOT = _path_from_env("ATLDSD_PROJECT_ROOT") or Path(__file__).resolve().parents[2]
PROJECT_ROOT = PROJECT_ROOT.expanduser().resolve()
SRC_ROOT = PROJECT_ROOT / "src"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

_ENV_VOCDEVKIT_PATH = _path_from_env("ATLDSD_VOCDEVKIT_PATH")
_ENV_DATASET_ROOT = _path_from_env("ATLDSD_DATASET_ROOT")
_VOCDEVKIT_CANDIDATES = [
    path
    for path in [
        _ENV_VOCDEVKIT_PATH,
        PROJECT_ROOT / "VOCdevkit",
        PROJECT_ROOT / "data" / "VOCdevkit",
        PROJECT_ROOT / "dataset" / "ATLDSD" / "VOCdevkit",
        PROJECT_ROOT / "datasets" / "ATLDSD" / "VOCdevkit",
        Path("/home/liuzhe/dataset/ATLDSD/VOCdevkit"),
        Path(r"D:\dataset\ATLDSD\VOCdevkit"),
    ]
    if path is not None
]
DEFAULT_VOCDEVKIT_PATH = _first_existing_path(_VOCDEVKIT_CANDIDATES, _VOCDEVKIT_CANDIDATES[0])
DEFAULT_DATASET_ROOT = (_ENV_DATASET_ROOT or DEFAULT_VOCDEVKIT_PATH.parent).expanduser()

LEGACY_MODELS_ROOT = SRC_ROOT / "models"
DEEPLABV3PLUS_ROOT = LEGACY_MODELS_ROOT / "deeplabv3plus"
SEGNEXT_ROOT = LEGACY_MODELS_ROOT / "segnext"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def output_path(*parts: str) -> Path:
    return OUTPUTS_ROOT.joinpath(*parts)


def resolve_voc_root(vocdevkit_path: Path | str = DEFAULT_VOCDEVKIT_PATH, year: str = "VOC2007") -> Path:
    """Return the VOC year directory used by the inherited training code."""

    path = Path(vocdevkit_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path / year
