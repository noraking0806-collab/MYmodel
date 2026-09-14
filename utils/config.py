"""Configuration and runtime helpers for the retained 2023HP pipeline."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch


MODEL_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = MODEL_ROOT.parent
MODEL_NAME = "2023HP"
DATASET_FOLDERS = {
    "adenocarcinoma": "EBHI-SEG/Adenocarcinoma",
    "glas": "Glas",
    "pglandseg": "PGlandSeg",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_SMALL8_URL = (
    "https://dl.fbaipublicfiles.com/dino/"
    "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
)
DINO_SMALL8_LOCAL = MODEL_ROOT / "dino_deitsmall8_300ep_pretrain.pth"


def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_FOLDERS:
        raise ValueError(f"Unknown dataset {name!r}; choose adenocarcinoma | glas | pglandseg")
    return key


def _resolve_device(value: str) -> torch.device:
    value = value.lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
