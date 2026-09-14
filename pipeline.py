"""HP baseline, isolated components, and selected combinations on gland splits.

This adapter keeps the released HP method (DINO-S/8 features, task-agnostic
and task-specific global hidden positives, the EMA task head, local hidden
positive mixing and an unsupervised cluster probe), while replacing the
repository's COCO/Cityscapes-only data layer and old training dependencies.

Training images are read from train.txt and their masks are never opened.
Validation masks are used only for early stopping and to resolve the arbitrary
two-cluster permutation. Test labels are used only to report metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    ColorJitter,
    GaussianBlur,
    InterpolationMode,
    RandomGrayscale,
    RandomResizedCrop,
)
from torchvision.transforms import functional as TF
from tqdm import tqdm


MODEL_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = MODEL_ROOT.parent
for import_root in (MODEL_ROOT, WORKSPACE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model.dino import vision_transformer as vits  # noqa: E402
from multi_receptive_field import MultiReceptiveFieldAdapter  # noqa: E402
from sparse_region_graph import SparseRegionGraphConsistency  # noqa: E402
from utils.layers import ClusterLookup  # noqa: E402
from utils.validation import (  # noqa: E402
    EarlyStopping,
    calculate_object_dice,
    format_metric_rows_for_csv,
    macro_binary_metrics_from_counts,
    network_metric_shape,
    resize_binary_for_metrics,
    validation_selection_key,
    verify_fixed_split,
)
from sage_metrics import calculate_metrics as sage_calculate_metrics  # noqa: E402


MODEL_NAME = "2023HP"
PIPELINE_MODEL_NAMES = {
    "none": MODEL_NAME,
    "sgc": "HP-SGC",
    "mrfa12": "HP-MRFA12",
    "mrfa12_sgc": "HP-MRFA12-SGC",
    "mrfa1_sgc": "HP-MRFA1-SGC",
    "mrfa2_sgc": "HP-MRFA2-SGC",
    "mrfa12_sgc_attr": "HP-MRFA12-SGC-ATTR",
    "mrfa12_sgc_rep": "HP-MRFA12-SGC-REP",
}
SGC_VARIANTS = frozenset(
    {
        "sgc",
        "mrfa12_sgc",
        "mrfa1_sgc",
        "mrfa2_sgc",
        "mrfa12_sgc_attr",
        "mrfa12_sgc_rep",
    }
)
MRFA_VARIANTS = frozenset(
    {
        "mrfa12",
        "mrfa12_sgc",
        "mrfa1_sgc",
        "mrfa2_sgc",
        "mrfa12_sgc_attr",
        "mrfa12_sgc_rep",
    }
)
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


def canonical_variant_name(name: str) -> str:
    key = name.strip().lower()
    if key not in PIPELINE_MODEL_NAMES:
        choices = " | ".join(PIPELINE_MODEL_NAMES)
        raise ValueError(f"Unknown HP variant {name!r}; choose {choices}")
    return key


def model_name_for_variant(variant: str) -> str:
    return PIPELINE_MODEL_NAMES[canonical_variant_name(variant)]


def variant_uses_sgc(variant: str) -> bool:
    return canonical_variant_name(variant) in SGC_VARIANTS


def variant_uses_mrfa(variant: str) -> bool:
    return canonical_variant_name(variant) in MRFA_VARIANTS


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


def _read_names(path: Path, max_samples: int | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation list not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if max_samples is not None:
        names = names[:max_samples]
    if not names:
        raise RuntimeError(f"Annotation list is empty: {path}")
    return names


def _resolve_png(directory: Path, entry: str) -> Path:
    path = directory / entry
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_pretrained_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in ("module.", "backbone.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


class GlandFiles:
    """Paths from the persistent comparison split, without implicit mask access."""

    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        split: str,
        max_samples: int | None = None,
    ) -> None:
        self.dataset = canonical_dataset_name(dataset)
        self.dataset_dir = Path(data_root) / DATASET_FOLDERS[self.dataset]
        image_subdir = "image" if self.dataset == "adenocarcinoma" else "images"
        label_subdir = "label" if self.dataset == "adenocarcinoma" else "labels"
        self.images_dir = self.dataset_dir / image_subdir
        self.labels_dir = self.dataset_dir / label_subdir
        self.annotation_dir = self.dataset_dir / "annotations"
        self.names = _read_names(self.annotation_dir / f"{split}.txt", max_samples)
        for name in self.names:
            _resolve_png(self.images_dir, name)

    def image_path(self, name: str) -> Path:
        return _resolve_png(self.images_dir, name)

    def label_path(self, name: str) -> Path:
        return _resolve_png(self.labels_dir, name)


class HPTrainDataset(Dataset[dict[str, Any]]):
    """Label-free training views matching the released HP augmentation policy."""

    def __init__(self, files: GlandFiles, size: int) -> None:
        self.files = files
        self.size = size
        self.color_jitter = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.grayscale = RandomGrayscale(p=0.2)
        self.blur = GaussianBlur(kernel_size=5)

    def __len__(self) -> int:
        return len(self.files.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.files.names[index]
        image = Image.open(self.files.image_path(name)).convert("RGB")
        top, left, height, width = RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(0.75, 4.0 / 3.0)
        )
        image = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [self.size, self.size],
            InterpolationMode.BILINEAR,
            antialias=True,
        )
        if random.random() < 0.5:
            image = TF.hflip(image)

        augmented = image.copy()
        augmented = self.color_jitter(augmented)
        augmented = self.grayscale(augmented)
        if random.random() < 0.5:
            augmented = self.blur(augmented)

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        augmented_tensor = TF.normalize(
            TF.to_tensor(augmented), IMAGENET_MEAN, IMAGENET_STD
        )
        return {"image": image_tensor, "image_aug": augmented_tensor, "name": name}


def _evaluation_tensor(image: Image.Image, size: int) -> torch.Tensor:
    resized = TF.resize(
        image, [size, size], InterpolationMode.BILINEAR, antialias=True
    )
    return TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)


class HPFeaturizer(nn.Module):
    """Device-safe implementation of the released HP DINO featurizer."""

    def __init__(
        self,
        dim: int,
        input_size: int,
        patch_size: int = 8,
        ema_m: float = 0.99,
        dropout: float = 0.1,
        dino_checkpoint: str | Path | None = None,
        load_pretrained: bool = True,
        allow_download: bool = True,
        variant: str = "none",
        context_bottleneck: int = 64,
        context_dilations: Iterable[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        if input_size % patch_size:
            raise ValueError("HP input size must be divisible by the DINO patch size")
        if patch_size != 8:
            raise ValueError("The released HP gland adapter supports DINO-S/8 only")

        self.dim = int(dim)
        self.variant = canonical_variant_name(variant)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.spatial_size = self.input_size // self.patch_size
        self.ema_m = float(ema_m)
        self.sigma = 28.0
        self.n_feats = 384

        self.model = vits.vit_small(patch_size=self.patch_size, num_classes=0)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
        if load_pretrained:
            self._load_dino_weights(dino_checkpoint, allow_download)

        self.context_adapter: nn.Module | None = None
        self.ema_context_adapter: nn.Module | None = None
        if variant_uses_mrfa(self.variant):
            self.context_adapter = MultiReceptiveFieldAdapter(
                channels=self.n_feats,
                bottleneck=context_bottleneck,
                dilations=tuple(context_dilations),
            )
            self.ema_context_adapter = copy.deepcopy(self.context_adapter)
            self.ema_context_adapter.eval()
            for parameter in self.ema_context_adapter.parameters():
                parameter.requires_grad = False

        self.dropout = nn.Dropout2d(p=dropout)
        self.cluster1 = nn.Conv2d(self.n_feats, self.dim, kernel_size=1)
        self.cluster2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.n_feats, self.dim, kernel_size=1),
        )
        self.ema_model1 = copy.deepcopy(self.cluster1)
        self.ema_model2 = copy.deepcopy(self.cluster2)
        for module in (self.ema_model1, self.ema_model2):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        patch_count = self.spatial_size * self.spatial_size
        neighbor_mask = torch.zeros((patch_count, patch_count), dtype=torch.bool)
        neighbor_count = torch.zeros((patch_count, 1), dtype=torch.float32)
        for row in range(self.spatial_size):
            for column in range(self.spatial_size):
                source = row * self.spatial_size + column
                neighbors: list[int] = []
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        target_row = row + delta_row
                        target_column = column + delta_column
                        if (
                            0 <= target_row < self.spatial_size
                            and 0 <= target_column < self.spatial_size
                        ):
                            neighbors.append(
                                target_row * self.spatial_size + target_column
                            )
                neighbor_mask[source, neighbors] = True
                neighbor_count[source] = float(len(neighbors))
        self.register_buffer("neighbor_mask", neighbor_mask, persistent=True)
        self.register_buffer("neighbor_count", neighbor_count, persistent=True)

    def _load_dino_weights(
        self,
        dino_checkpoint: str | Path | None,
        allow_download: bool,
    ) -> None:
        checkpoint = (
            Path(dino_checkpoint)
            if dino_checkpoint
            else DINO_SMALL8_LOCAL
        )
        if checkpoint.is_file():
            payload = _load_torch_file(checkpoint)
            source = str(checkpoint)
        elif dino_checkpoint:
            raise FileNotFoundError(f"DINO checkpoint not found: {checkpoint}")
        else:
            if not allow_download:
                raise FileNotFoundError(
                    "DINO-S/8 weights are required. Supply --dino_checkpoint or "
                    "remove --no_dino_download to allow the official download."
                )
            print("Downloading/loading the official DINO-S/8 pretrained weights...")
            payload = torch.hub.load_state_dict_from_url(
                DINO_SMALL8_URL, map_location="cpu", progress=True
            )
            source = DINO_SMALL8_URL

        if isinstance(payload, dict) and "teacher" in payload:
            state = payload["teacher"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported DINO checkpoint payload from {source}")
        state = _strip_pretrained_prefixes(state)
        result = self.model.load_state_dict(state, strict=True)
        print(f"Loaded DINO-S/8 weights from {source}: {result}")

    @torch.no_grad()
    def _update_ema(self) -> None:
        pairs: list[tuple[nn.Module, nn.Module]] = [
            (self.cluster1, self.ema_model1),
            (self.cluster2, self.ema_model2),
        ]
        if self.context_adapter is not None:
            assert self.ema_context_adapter is not None
            pairs.append((self.context_adapter, self.ema_context_adapter))
        for online, momentum in pairs:
            for source, target in zip(online.parameters(), momentum.parameters()):
                target.mul_(self.ema_m).add_(
                    source.detach(), alpha=1.0 - self.ema_m
                )
            for source, target in zip(online.buffers(), momentum.buffers()):
                target.copy_(source)

    def train(self, mode: bool = True) -> "HPFeaturizer":
        super().train(mode)
        self.model.eval()
        self.ema_model1.eval()
        self.ema_model2.eval()
        if self.ema_context_adapter is not None:
            self.ema_context_adapter.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.cluster1.parameters()
        yield from self.cluster2.parameters()
        if self.context_adapter is not None:
            yield from self.context_adapter.parameters()

    def forward(
        self,
        image: torch.Tensor,
        hp_train: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        self.model.eval()
        batch_size, _, height, width = image.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Input height and width must be divisible by patch size")
        feature_height = height // self.patch_size
        feature_width = width // self.patch_size
        if hp_train and (
            feature_height != self.spatial_size
            or feature_width != self.spatial_size
        ):
            raise ValueError(
                "HP local-positive training requires the configured square input size"
            )

        with torch.no_grad():
            if hp_train:
                feature_list, attention_list, _ = self.model.get_intermediate_feat(
                    image, n=1
                )
                tokens = feature_list[0]
                attention = attention_list[0]
            else:
                tokens = self.model.forward_feats(image)
                attention = None
            image_features = (
                tokens[:, 1:, :]
                .reshape(batch_size, feature_height, feature_width, self.n_feats)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        online_features = image_features
        if self.context_adapter is not None:
            online_features = self.context_adapter(online_features)
        dropped_features = self.dropout(online_features)
        linear_code = self.cluster1(dropped_features)
        nonlinear_code = self.cluster2(dropped_features)
        code = linear_code + nonlinear_code
        with torch.no_grad():
            teacher_features = image_features
            if self.ema_context_adapter is not None:
                teacher_features = self.ema_context_adapter(teacher_features)
            ema_linear = self.ema_model1(teacher_features)
            ema_nonlinear = self.ema_model2(teacher_features)
            ema_code = ema_linear + ema_nonlinear

        if not hp_train:
            return image_features, code, ema_code

        assert attention is not None
        with torch.no_grad():
            attention = attention[:, :, 1:, 1:].mean(dim=1).float()
            upper = torch.quantile(attention, 0.9, dim=2, keepdim=True)
            lower = torch.quantile(attention, 0.1, dim=2, keepdim=True)
            attention = torch.maximum(torch.minimum(attention, upper), lower)
            attention = attention.softmax(dim=-1) * self.sigma
            attention = attention.masked_fill(
                attention < attention.mean(dim=2, keepdim=True), 0.0
            )
            attention = attention * self.neighbor_mask.unsqueeze(0)

        flattened_code = code.flatten(2).transpose(1, 2)
        mixed_code = torch.bmm(attention.to(code.dtype), flattened_code)
        mixed_code = mixed_code / self.neighbor_count.to(code.dtype).unsqueeze(0)
        mixed_code = (
            mixed_code.transpose(1, 2)
            .reshape(batch_size, self.dim, feature_height, feature_width)
            .contiguous()
        )
        if self.training:
            self._update_ema()
        return (
            self.dropout(image_features),
            code,
            self.dropout(ema_code),
            self.dropout(mixed_code),
        )


def _cluster_logits(
    cluster_probe: ClusterLookup,
    code: torch.Tensor,
) -> torch.Tensor:
    clusters = F.normalize(cluster_probe.clusters, dim=1)
    features = F.normalize(code, dim=1)
    return torch.einsum("bchw,nc->bnhw", features, clusters)


def hidden_positive_loss(
    projected: torch.Tensor,
    task_specific: torch.Tensor,
    task_agnostic: torch.Tensor,
    pool_agnostic: torch.Tensor,
    pool_specific: torch.Tensor,
    projected_local: torch.Tensor,
    patches_per_image: int,
    temperature: float,
    rho: float,
    task_weight: float,
    reweighting: bool = True,
) -> torch.Tensor:
    """Device-safe formulation of the released GHP and LHP objective."""

    projected = F.normalize(projected.float(), dim=1)
    projected_local = F.normalize(projected_local.float(), dim=1)
    task_specific = F.normalize(task_specific.float(), dim=1)
    task_agnostic = F.normalize(task_agnostic.float(), dim=1)
    pool_agnostic = F.normalize(pool_agnostic.float(), dim=1)
    pool_specific = F.normalize(pool_specific.float(), dim=1)

    total_patches = projected.shape[0]
    if total_patches % patches_per_image:
        raise ValueError("Flattened HP features do not contain complete images")

    with torch.no_grad():
        threshold_agnostic = (task_agnostic @ pool_agnostic.T).amax(dim=1)
        threshold_specific = (task_specific @ pool_specific.T).amax(dim=1)

    image_losses: list[torch.Tensor] = []
    for start in range(0, total_patches, patches_per_image):
        stop = start + patches_per_image
        with torch.no_grad():
            similarity_agnostic = (
                task_agnostic[start:stop] @ task_agnostic.T
            )
            similarity_specific = (
                task_specific[start:stop] @ task_specific.T
            )
            positives_agnostic = (
                similarity_agnostic > threshold_agnostic[start:stop, None]
            ) | (similarity_agnostic > threshold_agnostic[None, :])
            positives_specific = (
                similarity_specific > threshold_specific[start:stop, None]
            ) | (similarity_specific > threshold_specific[None, :])
            row_indices = torch.arange(
                patches_per_image, device=projected.device
            )
            column_indices = torch.arange(start, stop, device=projected.device)
            positives_agnostic[row_indices, column_indices] = False
            positives_specific[row_indices, column_indices] = False

            sampled_negatives = (
                torch.rand(positives_agnostic.shape, device=projected.device)
                < rho
            )
            denominator_mask = positives_agnostic | sampled_negatives
            denominator_mask[row_indices, column_indices] = False
            positive_weights = positives_agnostic.float()
            positive_weights.add_(
                positives_specific.float(), alpha=float(task_weight)
            )
            # Every weighted positive must participate in the denominator.
            # This also prevents an empty denominator when only the gradually
            # introduced task-specific positives are available.
            denominator_mask |= positive_weights > 0
            valid = positive_weights.sum(dim=1) > 0

        if not bool(valid.any()):
            continue

        logits = (projected[start:stop] @ projected.T) / temperature
        log_denominator = torch.logsumexp(
            logits.masked_fill(~denominator_mask, float("-inf")), dim=1
        )
        log_probability = logits - log_denominator[:, None]
        weights = positive_weights[valid]
        global_loss = -(
            (weights * log_probability[valid]).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1e-6)
        )

        local_logits = (
            projected_local[start:stop] @ projected_local.T
        ) / temperature
        local_log_denominator = torch.logsumexp(
            local_logits.masked_fill(~denominator_mask, float("-inf")), dim=1
        )
        local_log_probability = local_logits - local_log_denominator[:, None]
        local_loss = -(
            (weights * local_log_probability[valid]).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1e-6)
        )

        if reweighting:
            positive_counts = weights.sum(dim=1)
            importance = positive_counts / positive_counts.mean().clamp_min(1e-6)
            global_loss = global_loss * importance
            local_loss = local_loss * importance
        image_losses.append(0.5 * (global_loss.mean() + local_loss.mean()))

    if not image_losses:
        return projected.sum() * 0.0
    return torch.stack(image_losses).mean()


@torch.inference_mode()
def build_reference_pools(
    model: HPFeaturizer,
    loader: DataLoader,
    dataset_size: int,
    pool_size: int,
    device: torch.device,
    amp: bool,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()
    pool_agnostic: list[torch.Tensor] = []
    pool_specific: list[torch.Tensor] = []
    remaining = pool_size
    images_to_sample = max(1, min(dataset_size, 64))
    patches_per_image = max(1, math.ceil(pool_size / images_to_sample))

    progress = tqdm(loader, desc=description, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            features, _, ema_code = model(images, hp_train=False)
        for batch_index in range(images.shape[0]):
            count = min(patches_per_image, remaining)
            patch_count = features.shape[-2] * features.shape[-1]
            indices = torch.randperm(patch_count, device=device)[:count]
            pool_agnostic.append(
                features[batch_index].flatten(1)[:, indices].T.float()
            )
            pool_specific.append(
                ema_code[batch_index].flatten(1)[:, indices].T.float()
            )
            remaining -= count
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    if not pool_agnostic:
        raise RuntimeError("Could not initialize HP reference pools")
    agnostic = torch.cat(pool_agnostic, dim=0)
    specific = torch.cat(pool_specific, dim=0)
    if agnostic.shape[0] < pool_size:
        repeat_indices = (
            torch.arange(pool_size - agnostic.shape[0], device=device)
            % agnostic.shape[0]
        )
        agnostic = torch.cat([agnostic, agnostic[repeat_indices]], dim=0)
        specific = torch.cat([specific, specific[repeat_indices]], dim=0)
    agnostic = F.normalize(agnostic[:pool_size], dim=1)
    specific = F.normalize(specific[:pool_size], dim=1)
    model.train(was_training)
    return agnostic, specific


def _mapping_scores(
    contingencies: list[np.ndarray],
    gland_cluster: int,
    cluster_maps: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict[str, Any]:
    other_cluster = 1 - gland_cluster
    counts = [
        (
            int(contingency[gland_cluster, 1]),
            int(contingency[gland_cluster, 0]),
            int(contingency[other_cluster, 1]),
            int(contingency[other_cluster, 0]),
        )
        for contingency in contingencies
    ]
    object_dice_values = [
        calculate_object_dice(clusters == gland_cluster, target)
        for clusters, target in zip(cluster_maps, targets)
    ]
    return macro_binary_metrics_from_counts(counts, object_dice_values)


@torch.inference_mode()
def evaluate_validation_mapping(
    model: HPFeaturizer,
    cluster_probe: ClusterLookup,
    files: GlandFiles,
    input_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[int, dict[str, Any]]:
    model.eval()
    cluster_probe.eval()
    contingencies: list[np.ndarray] = []
    cluster_maps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for name in tqdm(files.names, desc="HP validation", leave=False):
        image = Image.open(files.image_path(name)).convert("RGB")
        target = np.asarray(Image.open(files.label_path(name))) > 0
        inputs = _evaluation_tensor(image, input_size)[None].to(device)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            _, code, _ = model(inputs, hp_train=False)
            logits = _cluster_logits(cluster_probe, code)
            logits = F.interpolate(
                logits,
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )
        clusters = logits.argmax(dim=1)[0].cpu().numpy()
        cluster_maps.append(clusters)
        targets.append(target)
        contingency = np.zeros((2, 2), dtype=np.int64)
        for cluster_index in (0, 1):
            selected = clusters == cluster_index
            contingency[cluster_index, 0] += int(
                np.logical_and(selected, ~target).sum()
            )
            contingency[cluster_index, 1] += int(
                np.logical_and(selected, target).sum()
            )
        contingencies.append(contingency)

    candidates: list[tuple[int, dict[str, Any]]] = []
    for index in (0, 1):
        metrics = _mapping_scores(contingencies, index, cluster_maps, targets)
        candidates.append((index, metrics))
    gland_cluster, metrics = max(
        candidates, key=lambda item: validation_selection_key(item[1])
    )
    return gland_cluster, metrics


def _component_hparams(args: argparse.Namespace) -> dict[str, Any]:
    component_hparams: dict[str, Any] = {}
    if variant_uses_sgc(args.variant):
        component_hparams.update(
            {
                "graph_weight": args.graph_weight,
                "graph_grid_size": args.graph_grid_size,
                "graph_neighbors": args.graph_neighbors,
                "graph_attraction_weight": getattr(
                    args, "graph_attraction_weight", 1.0
                ),
                "graph_repulsion_weight": args.graph_repulsion_weight,
                "graph_negative_quantile": args.graph_negative_quantile,
                "graph_prediction_temperature": args.graph_prediction_temperature,
                "graph_negative_margin": args.graph_negative_margin,
            }
        )
    if variant_uses_mrfa(args.variant):
        component_hparams.update(
            {
                "context_bottleneck": args.context_bottleneck,
                "context_dilations": list(args.context_dilations),
            }
        )
    return component_hparams


def _checkpoint_payload(
    model: HPFeaturizer,
    project_head: nn.Module,
    cluster_probe: ClusterLookup,
    optimizer: torch.optim.Optimizer,
    cluster_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset: str,
    epoch: int,
    global_step: int,
    gland_cluster: int,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    variant = canonical_variant_name(args.variant)
    payload = {
        "model": model.state_dict(),
        "project_head": project_head.state_dict(),
        "cluster_probe": cluster_probe.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cluster_optimizer": cluster_optimizer.state_dict(),
        "model_name": model_name_for_variant(variant),
        "pipeline_name": model_name_for_variant(variant),
        "variant": variant,
        "component_hparams": _component_hparams(args),
        "dataset": dataset,
        "epoch": epoch,
        "global_step": global_step,
        "gland_cluster": gland_cluster,
        "validation_metrics": validation_metrics,
        "input_size": args.input_size,
        "patch_size": args.patch_size,
        "dim": args.dim,
        "ema_m": args.ema_m,
        "num_classes": 2,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "rho": args.rho,
        "pool_size": args.pool_size,
        "learning_rate": args.learning_rate,
        "cluster_learning_rate": args.cluster_learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    return payload


_RESTORABLE_MODEL_STATE_ROOTS = frozenset(
    {
        "cluster1",
        "cluster2",
        "ema_model1",
        "ema_model2",
        "context_adapter",
        "ema_context_adapter",
    }
)


def _clone_to_cpu(value: Any) -> Any:
    """Clone a nested state object without retaining accelerator storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _capture_validation_candidate_state(
    model: nn.Module,
    project_head: nn.Module,
    cluster_probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    cluster_optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Capture only the state that can change during HP training.

    The frozen DINO backbone and fixed neighbourhood buffers are identical for
    every epoch, so retaining them for every validation candidate would add
    roughly 87 MB per candidate without changing which epoch can be restored.
    """

    model_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
        if key.split(".", 1)[0] in _RESTORABLE_MODEL_STATE_ROOTS
    }
    return {
        "model": model_state,
        "project_head": _clone_to_cpu(project_head.state_dict()),
        "cluster_probe": _clone_to_cpu(cluster_probe.state_dict()),
        "optimizer": _clone_to_cpu(optimizer.state_dict()),
        "cluster_optimizer": _clone_to_cpu(cluster_optimizer.state_dict()),
    }


def _restore_validation_candidate_state(
    state: dict[str, Any],
    model: nn.Module,
    project_head: nn.Module,
    cluster_probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    cluster_optimizer: torch.optim.Optimizer,
) -> None:
    """Restore a candidate before building the final, complete checkpoint."""

    load_result = model.load_state_dict(state["model"], strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in retained HP model state: "
            f"{load_result.unexpected_keys}"
        )
    missing_restorable = [
        key
        for key in load_result.missing_keys
        if key.split(".", 1)[0] in _RESTORABLE_MODEL_STATE_ROOTS
    ]
    if missing_restorable:
        raise RuntimeError(
            "Retained HP model state is incomplete: "
            f"{missing_restorable}"
        )
    project_head.load_state_dict(state["project_head"], strict=True)
    cluster_probe.load_state_dict(state["cluster_probe"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    cluster_optimizer.load_state_dict(state["cluster_optimizer"])


def _retained_validation_candidate_indices(
    candidates: list[dict[str, Any]],
    tolerance: float,
) -> set[int]:
    """Return candidates whose states could still win after a future epoch.

    A candidate below the current Mean-Dice window can never re-enter because
    the historical best only increases. A Pareto-dominated candidate can also
    never win: its dominator has at least as much Mean Dice and Object Dice.
    Metrics for *all* epochs remain in ``EarlyStopping`` and the JSON history;
    this function only bounds temporary checkpoint-state storage.
    """

    if not candidates:
        return set()
    best_mdice = max(float(candidate["mdice"]) for candidate in candidates)
    threshold = best_mdice - float(tolerance)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["mdice"]) >= threshold
    ]
    retained: set[int] = set()
    for candidate in eligible:
        candidate_mdice = float(candidate["mdice"])
        candidate_object = float(candidate["object_dice"])
        dominated = any(
            (
                float(other["mdice"]) >= candidate_mdice
                and float(other["object_dice"]) >= candidate_object
                and (
                    float(other["mdice"]) > candidate_mdice
                    or float(other["object_dice"]) > candidate_object
                )
            )
            for other in eligible
        )
        if not dominated:
            retained.add(int(candidate["candidate_index"]))
    return retained


def _checkpoint_selection_metadata(
    stopper: EarlyStopping,
) -> dict[str, Any]:
    return {
        "primary_metric": "mdice",
        "primary_tolerance": stopper.primary_tolerance,
        "eligibility_rule": (
            "mdice >= best_validation_mdice_observed - primary_tolerance"
        ),
        "secondary_metric": "object_dice",
        "secondary_metric_definition": (
            "classic_glas_bidirectional_area_weighted_8_connected"
        ),
        "tie_breaker": "mdice",
        "selection_scope": "all_observed_validation_epochs",
    }


def train(args: argparse.Namespace) -> Path:
    variant = canonical_variant_name(args.variant)
    model_name = model_name_for_variant(variant)
    dataset = canonical_dataset_name(args.dataset)
    dataset_dir = Path(args.data_root) / DATASET_FOLDERS[dataset]
    split_metadata = verify_fixed_split(dataset_dir / "annotations")
    train_files = GlandFiles(args.data_root, dataset, "train", args.max_samples)
    val_files = GlandFiles(
        args.data_root, dataset, "val", args.max_val_samples
    )
    output = (
        Path(args.output_dir)
        if args.output_dir
        else WORKSPACE_ROOT / "train_out" / model_name / dataset
    )
    output.mkdir(parents=True, exist_ok=True)

    dataset_object = HPTrainDataset(train_files, args.input_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.resolved_device.type == "cuda",
        drop_last=len(dataset_object) >= args.batch_size,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    memory_loader = DataLoader(
        dataset_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.resolved_device.type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError("HP training loader is empty")

    model = HPFeaturizer(
        dim=args.dim,
        input_size=args.input_size,
        patch_size=args.patch_size,
        ema_m=args.ema_m,
        dino_checkpoint=args.dino_checkpoint,
        load_pretrained=True,
        allow_download=not args.no_dino_download,
        variant=variant,
        context_bottleneck=args.context_bottleneck,
        context_dilations=args.context_dilations,
    ).to(args.resolved_device)
    project_head = nn.Linear(args.dim, args.dim).to(args.resolved_device)
    cluster_probe = ClusterLookup(args.dim, 2).to(args.resolved_device)
    region_graph = (
        SparseRegionGraphConsistency(
            grid_size=args.graph_grid_size,
            neighbors=args.graph_neighbors,
            repulsion_weight=args.graph_repulsion_weight,
            negative_quantile=args.graph_negative_quantile,
            prediction_temperature=args.graph_prediction_temperature,
            negative_margin=args.graph_negative_margin,
            attraction_weight=args.graph_attraction_weight,
        ).to(args.resolved_device)
        if variant_uses_sgc(variant)
        else None
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.trainable_parameters())},
            {"params": project_head.parameters()},
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    cluster_optimizer = torch.optim.Adam(
        cluster_probe.parameters(), lr=args.cluster_learning_rate
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and args.resolved_device.type == "cuda",
    )
    stopper = EarlyStopping(
        patience=args.patience, min_delta=args.min_delta
    )
    total_steps = args.epochs * len(loader)
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps >= 0
        else min(1000, max(1, round(total_steps * 0.1)))
    )

    pool_agnostic, pool_specific = build_reference_pools(
        model,
        memory_loader,
        len(dataset_object),
        args.pool_size,
        args.resolved_device,
        args.amp,
        "HP initial reference pool",
    )

    global_step = 0
    validation_history: list[dict[str, Any]] = []
    candidate_state_paths: dict[int, Path] = {}
    candidate_directory = tempfile.TemporaryDirectory(
        prefix=".validation_candidates_",
        dir=output,
    )
    candidate_root = Path(candidate_directory.name)
    checkpoint = output / "checkpoint_final.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        project_head.train()
        cluster_probe.train()
        losses: list[float] = []
        progress = tqdm(loader, desc=f"{model_name} epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(
                args.resolved_device, non_blocking=True
            )
            images_augmented = batch["image_aug"].to(
                args.resolved_device, non_blocking=True
            )
            if (
                global_step > 0
                and args.renew_interval > 0
                and global_step % args.renew_interval == 0
            ):
                pool_agnostic, pool_specific = build_reference_pools(
                    model,
                    memory_loader,
                    len(dataset_object),
                    args.pool_size,
                    args.resolved_device,
                    args.amp,
                    "HP renewed reference pool",
                )
                model.train()

            if global_step <= warmup_steps:
                task_weight = 0.0
            else:
                task_weight = (global_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                task_weight = min(1.0, max(0.0, task_weight))

            optimizer.zero_grad(set_to_none=True)
            cluster_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=args.resolved_device.type,
                enabled=args.amp and args.resolved_device.type == "cuda",
            ):
                features, code, ema_code, local_code = model(
                    images, hp_train=True
                )
                _, augmented_code, _ = model(
                    images_augmented, hp_train=False
                )

                flattened_features = (
                    features.permute(0, 2, 3, 1)
                    .reshape(-1, features.shape[1])
                )
                flattened_code = (
                    code.permute(0, 2, 3, 1).reshape(-1, args.dim)
                )
                flattened_ema = (
                    ema_code.permute(0, 2, 3, 1).reshape(-1, args.dim)
                )
                flattened_local = (
                    local_code.permute(0, 2, 3, 1)
                    .reshape(-1, args.dim)
                )
                flattened_augmented = (
                    augmented_code.permute(0, 2, 3, 1)
                    .reshape(-1, args.dim)
                )

                projected = F.normalize(
                    project_head(flattened_code), dim=1
                )
                projected_local = F.normalize(
                    project_head(flattened_local), dim=1
                )
                projected_augmented = F.normalize(
                    project_head(flattened_augmented), dim=1
                )
                consistency = torch.linalg.vector_norm(
                    projected - projected_augmented, dim=1
                ).mean()
                hp_loss = hidden_positive_loss(
                    projected,
                    flattened_ema,
                    flattened_features,
                    pool_agnostic,
                    pool_specific,
                    projected_local,
                    patches_per_image=(
                        model.spatial_size * model.spatial_size
                    ),
                    temperature=args.temperature,
                    rho=args.rho,
                    task_weight=task_weight,
                    reweighting=bool(args.reweighting),
                )
                cluster_loss, _ = cluster_probe(
                    code.detach(), None, is_direct=False
                )
                loss = hp_loss + args.alpha * consistency + cluster_loss
                graph_loss: torch.Tensor | None = None
                if region_graph is not None:
                    graph_loss, _ = region_graph(
                        code, ema_code, cluster_probe.clusters
                    )
                    loss = (
                        loss
                        + args.graph_weight
                        * task_weight
                        * graph_loss
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.trainable_parameters())
                + list(project_head.parameters()),
                args.grad_norm,
            )
            scaler.step(optimizer)
            scaler.step(cluster_optimizer)
            scaler.update()
            global_step += 1

            losses.append(float(loss.detach()))
            postfix = {
                "loss": f"{np.mean(losses):.4f}",
                "hp": f"{float(hp_loss.detach()):.4f}",
                "lambda_hp": f"{task_weight:.3f}",
            }
            if graph_loss is not None:
                postfix["graph"] = f"{float(graph_loss.detach()):.4f}"
            progress.set_postfix(**postfix)

        gland_cluster, validation_metrics = evaluate_validation_mapping(
            model,
            cluster_probe,
            val_files,
            args.input_size,
            args.resolved_device,
            args.amp,
        )
        candidate_index = len(validation_history)
        selection_changed = stopper.update(
            validation_metrics["mdice"],
            validation_metrics["object_dice"],
        )
        validation_history.append(
            {
                "candidate_index": candidate_index,
                "epoch": epoch,
                "global_step": global_step,
                "gland_cluster": int(gland_cluster),
                "mdice": float(validation_metrics["mdice"]),
                "gland_dice": float(validation_metrics["gland_dice"]),
                "object_dice": float(validation_metrics["object_dice"]),
                "validation_metrics": dict(validation_metrics),
            }
        )
        assert stopper.candidates is not None
        retained_indices = _retained_validation_candidate_indices(
            stopper.candidates,
            stopper.primary_tolerance,
        )
        if candidate_index in retained_indices:
            candidate_path = (
                candidate_root / f"candidate_{candidate_index:04d}.pth"
            )
            torch.save(
                _capture_validation_candidate_state(
                    model,
                    project_head,
                    cluster_probe,
                    optimizer,
                    cluster_optimizer,
                ),
                candidate_path,
            )
            candidate_state_paths[candidate_index] = candidate_path
        for stale_index in set(candidate_state_paths) - retained_indices:
            stale_path = candidate_state_paths.pop(stale_index)
            stale_path.unlink(missing_ok=True)
        print(
            f"[HP validation] epoch={epoch} "
            f"mDice={100 * validation_metrics['mdice']:.2f}% "
            f"glandDice={100 * validation_metrics['gland_dice']:.2f}% "
            f"ObjectDice={100 * validation_metrics['object_dice']:.2f}% "
            f"gland_cluster={gland_cluster} "
            f"selection_changed={selection_changed} "
            f"selected_candidate={stopper.selected_index}"
        )
        if stopper.should_stop:
            print(f"[HP] Early stopping after epoch {epoch}.")
            break

    try:
        if stopper.selected_index is None or not validation_history:
            raise RuntimeError("HP training did not record validation metrics")
        selected_index = stopper.selected_index
        selected_path = candidate_state_paths.get(selected_index)
        if selected_path is None or not selected_path.is_file():
            raise RuntimeError(
                "The globally selected validation candidate has no retained "
                f"state: candidate {selected_index}"
            )
        _restore_validation_candidate_state(
            _load_torch_file(selected_path),
            model,
            project_head,
            cluster_probe,
            optimizer,
            cluster_optimizer,
        )
        selected_record = validation_history[selected_index]
        best_epoch = int(selected_record["epoch"])
        best_validation_metrics = dict(
            selected_record["validation_metrics"]
        )
        selection_metadata = _checkpoint_selection_metadata(stopper)
        checkpoint_payload = _checkpoint_payload(
            model,
            project_head,
            cluster_probe,
            optimizer,
            cluster_optimizer,
            args,
            dataset,
            best_epoch,
            int(selected_record["global_step"]),
            int(selected_record["gland_cluster"]),
            best_validation_metrics,
        )
        checkpoint_payload.update(
            {
                "best_validation_mdice_observed": (
                    stopper.best_primary_observed
                ),
                "validation_mdice_tolerance": stopper.primary_tolerance,
                "selected_validation_mdice_threshold": (
                    stopper.best_primary_observed
                    - stopper.primary_tolerance
                ),
                "validation_candidates": [
                    dict(candidate)
                    for candidate in (stopper.candidates or [])
                ],
                "checkpoint_selection": selection_metadata,
            }
        )
        torch.save(checkpoint_payload, checkpoint)
    finally:
        candidate_directory.cleanup()

    final_mdice_threshold = (
        stopper.best_primary_observed - stopper.primary_tolerance
    )
    for record in validation_history:
        record["eligible_in_final_mdice_window"] = bool(
            float(record["mdice"]) >= final_mdice_threshold
        )
        record["selected_checkpoint"] = bool(
            int(record["candidate_index"]) == selected_index
        )
    history_payload = {
        "checkpoint_selection": selection_metadata,
        "best_validation_mdice_observed": stopper.best_primary_observed,
        "final_mdice_threshold": final_mdice_threshold,
        "selected_candidate_index": selected_index,
        "selected_epoch": best_epoch,
        "epochs": validation_history,
    }
    (output / "validation_history.json").write_text(
        json.dumps(history_payload, indent=2), encoding="utf-8"
    )
    training_parameters = {
        "input_size": args.input_size,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "dim": args.dim,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "rho": args.rho,
        "ema_m": args.ema_m,
        "pool_size": args.pool_size,
        "renew_interval": args.renew_interval,
        "warmup_steps": warmup_steps,
        "learning_rate": args.learning_rate,
        "cluster_learning_rate": args.cluster_learning_rate,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
    }
    component_hparams = _component_hparams(args)
    if component_hparams:
        training_parameters["component_hparams"] = component_hparams
    summary = {
        "model": model_name,
        "variant": variant,
        "dataset": dataset,
        "training_images": len(train_files.names),
        "validation_images": len(val_files.names),
        "labels_used_for_optimization": False,
        "validation_labels_used_for_cluster_permutation_and_early_stopping": True,
        "best_epoch": best_epoch,
        "best_validation_mdice": best_validation_metrics["mdice"],
        "best_validation_gland_dice": best_validation_metrics["gland_dice"],
        "best_validation_object_dice": best_validation_metrics["object_dice"],
        "best_validation_mdice_observed": stopper.best_primary_observed,
        "validation_mdice_tolerance": stopper.primary_tolerance,
        "checkpoint_selection": selection_metadata,
        "checkpoint": str(checkpoint),
        "validation_history": str(output / "validation_history.json"),
        "parameters": training_parameters,
        "fixed_split_metadata": split_metadata,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"HP training complete: {checkpoint}")
    gc.collect()
    if args.resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compatibility wrapper for the shared SAGE metric implementation."""
    return sage_calculate_metrics(prediction, target)


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    selected = mask.astype(bool)
    output[selected] = (
        0.55 * output[selected]
        + 0.45 * np.asarray(color, np.float32)
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def save_visual(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    max_side: int,
) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    if scale < 1.0:
        image = cv2.resize(
            image, size, interpolation=cv2.INTER_AREA
        )
        target = cv2.resize(
            target.astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    panels = [
        (image, "Image"),
        (_overlay(image, target, (0, 220, 0)), "Ground truth"),
        (
            _overlay(image, prediction, (255, 70, 40)),
            "Prediction",
        ),
    ]
    labelled: list[np.ndarray] = []
    for panel, label in panels:
        bar = np.zeros((30, panel.shape[1], 3), np.uint8)
        cv2.putText(
            bar,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labelled.append(np.concatenate([bar, panel], axis=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate(labelled, axis=1)
    if not cv2.imwrite(
        str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"Could not write visualization: {path}")


def _mean_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    keys = [key for key in rows[0] if key != "sample"]
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def _summary(
    dataset: str,
    checkpoint: Path,
    count: int,
    metrics: dict[str, float],
    model_name: str = MODEL_NAME,
    variant: str = "none",
) -> str:
    percent = [
            ("Pixel Accuracy", "pixel_accuracy"),
            ("Background IoU", "background_iou"),
            ("Gland IoU", "gland_iou"),
            ("Mean IoU", "mean_iou"),
            ("Background Dice", "background_dice"),
            ("Gland Dice", "gland_dice"),
            ("Mean Dice", "mean_dice"),
            ("Object Dice", "object_dice"),
        ]
    module_description = {
        "none": "baseline",
        "sgc": "sparse region-graph consistency",
        "mrfa12": "two-radius multi-receptive-field adapter",
        "mrfa12_sgc": (
            "two-radius multi-receptive-field adapter + sparse "
            "region-graph consistency"
        ),
        "mrfa1_sgc": (
            "dilation-1 receptive-field adapter + sparse "
            "region-graph consistency"
        ),
        "mrfa2_sgc": (
            "dilation-2 receptive-field adapter + sparse "
            "region-graph consistency"
        ),
        "mrfa12_sgc_attr": (
            "two-radius multi-receptive-field adapter + attraction-only "
            "sparse region-graph consistency"
        ),
        "mrfa12_sgc_rep": (
            "two-radius multi-receptive-field adapter + repulsion-only "
            "sparse region-graph consistency"
        ),
    }[canonical_variant_name(variant)]
    lines = [
        f"{model_name} Test Summary",
        "=" * 40,
        f"Dataset: {dataset.upper()}",
        (
            "Model: HP (DINO-S/8 + hidden positives + "
            f"cluster probe + {module_description}, unsupervised)"
        ),
        f"Checkpoint: {checkpoint}",
        f"Samples: {count}",
        "",
    ]
    lines.extend(
        f"{label:<20} {100 * metrics[key]:9.2f}%"
        for label, key in percent
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def test(args: argparse.Namespace) -> dict[str, float]:
    dataset = canonical_dataset_name(args.dataset)
    dataset_dir = Path(args.data_root) / DATASET_FOLDERS[dataset]
    verify_fixed_split(dataset_dir / "annotations")
    files = GlandFiles(
        args.data_root, dataset, args.split, args.max_samples
    )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = _load_torch_file(checkpoint)
    checkpoint_variant = canonical_variant_name(
        str(payload.get("variant", "none"))
    )
    requested_variant = canonical_variant_name(args.variant)
    if requested_variant != checkpoint_variant:
        raise ValueError(
            f"Requested variant {requested_variant!r} does not match "
            f"checkpoint variant {checkpoint_variant!r}"
        )
    model_name = model_name_for_variant(checkpoint_variant)
    checkpoint_name = payload.get(
        "pipeline_name", payload.get("model_name")
    )
    if checkpoint_name != model_name:
        raise ValueError(
            f"Checkpoint is not a {model_name} checkpoint: {checkpoint}"
        )
    if canonical_dataset_name(str(payload.get("dataset"))) != dataset:
        raise ValueError(
            f"Checkpoint dataset {payload.get('dataset')!r} "
            f"does not match {dataset!r}"
        )

    input_size = int(payload["input_size"])
    component_hparams = dict(payload.get("component_hparams", {}))
    model = HPFeaturizer(
        dim=int(payload["dim"]),
        input_size=input_size,
        patch_size=int(payload["patch_size"]),
        ema_m=float(payload["ema_m"]),
        load_pretrained=False,
        variant=checkpoint_variant,
        context_bottleneck=int(
            component_hparams.get("context_bottleneck", 64)
        ),
        context_dilations=tuple(
            component_hparams.get("context_dilations", (1, 2, 3))
        ),
    ).to(args.resolved_device)
    cluster_probe = ClusterLookup(
        int(payload["dim"]), 2
    ).to(args.resolved_device)
    model.load_state_dict(payload["model"], strict=True)
    cluster_probe.load_state_dict(
        payload["cluster_probe"], strict=True
    )
    model.eval()
    cluster_probe.eval()
    gland_cluster = int(payload["gland_cluster"])

    output = (
        Path(args.output_dir)
        if args.output_dir
        else WORKSPACE_ROOT / "test_out" / model_name / dataset
    )
    visuals = output / "visuals"
    predictions = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in tqdm(files.names, desc=f"HP {args.split}"):
        image = Image.open(files.image_path(name)).convert("RGB")
        image_np = np.asarray(image)
        target = np.asarray(
            Image.open(files.label_path(name))
        ) > 0
        inputs = _evaluation_tensor(
            image, input_size
        )[None].to(args.resolved_device)
        with torch.amp.autocast(
            device_type=args.resolved_device.type,
            enabled=(
                args.amp
                and args.resolved_device.type == "cuda"
            ),
        ):
            _, code, _ = model(inputs, hp_train=False)
            logits = _cluster_logits(cluster_probe, code)
            logits = F.interpolate(
                logits,
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )
        clusters = logits.argmax(dim=1)[0].cpu().numpy()
        prediction = clusters == gland_cluster
        metric_prediction, metric_target = prediction, target
        if args.metric_resolution == "network":
            metric_prediction, metric_target = resize_binary_for_metrics(
                prediction,
                target,
                network_metric_shape(target.shape, input_size),
            )
        metrics = sage_calculate_metrics(metric_prediction, metric_target)
        rows.append({"sample": name, **metrics})
        if not cv2.imwrite(
            str(predictions / f"{Path(name).stem}.png"),
            prediction.astype(np.uint8) * 255,
        ):
            raise OSError(f"Could not write prediction for {name}")
        save_visual(
            image_np,
            target,
            prediction,
            visuals / f"{name}.png",
            args.visual_max_side,
        )

    means = _mean_metrics(rows)
    with (output / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(format_metric_rows_for_csv(rows))
        writer.writerow(
            format_metric_rows_for_csv([{"sample": "MEAN", **means}])[0]
        )
    summary = _summary(
        dataset,
        checkpoint,
        len(rows),
        means,
        model_name=model_name,
        variant=checkpoint_variant,
    )
    (output / "summary.txt").write_text(
        summary, encoding="utf-8"
    )
    print(summary)
    return means


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HP gland-segmentation component and combination pipelines"
        )
    )
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_FOLDERS),
        required=True,
    )
    parser.add_argument(
        "--data_root", default=r"D:\1.KAN\data"
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument(
        "--variant",
        choices=sorted(PIPELINE_MODEL_NAMES),
        default="none",
        help="Select the HP baseline, an isolated component, or a combination",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument(
        "--temperature", type=float, default=0.8
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.02)
    parser.add_argument("--ema_m", type=float, default=0.99)
    parser.add_argument("--pool_size", type=int, default=2048)
    parser.add_argument(
        "--renew_interval", type=int, default=100
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        help=(
            "-1 scales the paper warm-up to 10 percent "
            "of this dataset run"
        ),
    )
    parser.add_argument(
        "--reweighting",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4
    )
    parser.add_argument(
        "--cluster_learning_rate",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1
    )
    parser.add_argument("--graph_weight", type=float, default=0.10)
    parser.add_argument("--graph_grid_size", type=int, default=6)
    parser.add_argument("--graph_neighbors", type=int, default=4)
    parser.add_argument(
        "--graph_attraction_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--graph_repulsion_weight", type=float, default=0.50
    )
    parser.add_argument(
        "--graph_negative_quantile", type=float, default=0.25
    )
    parser.add_argument(
        "--graph_prediction_temperature", type=float, default=0.20
    )
    parser.add_argument(
        "--graph_negative_margin", type=float, default=0.25
    )
    parser.add_argument("--context_bottleneck", type=int, default=64)
    parser.add_argument(
        "--context_dilations",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--grad_norm", type=float, default=10.0
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--min_delta", type=float, default=0.0001
    )
    parser.add_argument("--dino_checkpoint", default=None)
    parser.add_argument(
        "--no_dino_download", action="store_true"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=None
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--visual_max_side", type=int, default=512
    )
    parser.add_argument(
        "--metric_resolution",
        choices=["network", "original"],
        default="original",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> Any:
    args = build_parser().parse_args(
        list(argv) if argv is not None else None
    )
    if (
        args.input_size <= 0
        or args.input_size % args.patch_size
    ):
        raise ValueError(
            "--input_size must be positive and divisible "
            "by --patch_size"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.pool_size <= 0:
        raise ValueError("--pool_size must be positive")
    if not 0.0 < args.temperature:
        raise ValueError("--temperature must be positive")
    if not 0.0 <= args.rho <= 1.0:
        raise ValueError("--rho must be in [0, 1]")
    if not 0.0 <= args.ema_m < 1.0:
        raise ValueError("--ema_m must be in [0, 1)")
    if (
        args.graph_weight < 0.0
        or args.graph_attraction_weight < 0.0
        or args.graph_repulsion_weight < 0.0
    ):
        raise ValueError("Graph loss weights must be non-negative")
    if args.graph_grid_size <= 1 or args.graph_neighbors <= 0:
        raise ValueError("Graph grid and neighbor count must be positive")
    if not 0.0 < args.graph_negative_quantile <= 1.0:
        raise ValueError("--graph_negative_quantile must be in (0, 1]")
    if args.graph_prediction_temperature <= 0.0:
        raise ValueError("--graph_prediction_temperature must be positive")
    if not 0.0 <= args.graph_negative_margin <= 1.0:
        raise ValueError("--graph_negative_margin must be in [0, 1]")
    if args.context_bottleneck <= 0 or any(
        dilation <= 0 for dilation in args.context_dilations
    ):
        raise ValueError("Context bottleneck and dilations must be positive")
    args.resolved_device = _resolve_device(args.device)
    _seed_everything(args.seed)
    if args.mode == "train":
        return train(args)
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in test mode")
    return test(args)


if __name__ == "__main__":
    main()
