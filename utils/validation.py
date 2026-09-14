"""Shared validation and early-stopping utilities for unified pipelines.

Validation predictions are restored to each sample's original Ground Truth
resolution. Foreground/gland Dice, background Dice, mDice and Object Dice are
calculated per image, then averaged with equal image weight. Checkpoints must
remain within 0.5 percentage points of the best observed macro mDice; among
eligible checkpoints, macro Object Dice selects the retained state.
Micro/global Dice is diagnostic and never selects mappings or checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


FIXED_SPLIT_SEED = 42
FIXED_TRAIN_FRACTION = 0.70
FIXED_VALIDATION_FRACTION = 0.10
FIXED_TEST_FRACTION = 0.20
TRAIN_VAL_ONLY_TRAIN_FRACTION = 0.80
TRAIN_VAL_ONLY_VALIDATION_FRACTION = 0.20
MDICE_CANDIDATE_TOLERANCE = 0.005


def _dice_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    # Match SAGE's calculate_dice_coefficient smooth=1e-5 exactly, while
    # retaining the explicit empty-class convention used by the validator.
    smooth = 1e-5
    return (
        1.0
        if denominator == 0
        else (2.0 * int(tp) + smooth) / (denominator + smooth)
    )


def binary_confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return foreground ``tp, fp, fn, tn`` for one original-size image."""

    prediction = np.asarray(prediction).astype(bool, copy=False)
    target = np.asarray(target).astype(bool, copy=False)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Validation prediction and target must both be two-dimensional"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "Original-resolution validation requires matching prediction and "
            f"target shapes, got {prediction.shape} and {target.shape}"
        )
    return (
        int(np.logical_and(prediction, target).sum()),
        int(np.logical_and(prediction, ~target).sum()),
        int(np.logical_and(~prediction, target).sum()),
        int(np.logical_and(~prediction, ~target).sum()),
    )


def calculate_object_dice(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return the classic GlaS area-weighted, bidirectional Object Dice.

    Each ground-truth component is paired with the prediction having the
    largest pixel overlap, and vice versa.  The paired region Dice scores are
    area weighted in each direction and the two directional scores are
    averaged.  GlaS uses 8-connected components for two-dimensional masks.
    """

    prediction = np.asarray(prediction).astype(bool, copy=False)
    target = np.asarray(target).astype(bool, copy=False)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("Object Dice requires two-dimensional masks")
    if prediction.shape != target.shape:
        raise ValueError(
            "Object Dice requires matching prediction and target shapes, "
            f"got {prediction.shape} and {target.shape}"
        )

    connectivity = np.ones((3, 3), dtype=np.uint8)
    pred_labeled, num_predictions = label(
        prediction, structure=connectivity
    )
    target_labeled, num_targets = label(target, structure=connectivity)
    if num_targets == 0 or num_predictions == 0:
        return 1.0 if num_targets == num_predictions else 0.0

    pred_areas = np.bincount(
        pred_labeled.ravel(), minlength=num_predictions + 1
    ).astype(np.float64, copy=False)
    target_areas = np.bincount(
        target_labeled.ravel(), minlength=num_targets + 1
    ).astype(np.float64, copy=False)

    overlap = np.logical_and(prediction, target)
    best_overlap_for_target = np.zeros(num_targets + 1, dtype=np.int64)
    best_pred_for_target = np.zeros(num_targets + 1, dtype=np.int64)
    best_overlap_for_pred = np.zeros(num_predictions + 1, dtype=np.int64)
    best_target_for_pred = np.zeros(num_predictions + 1, dtype=np.int64)
    if np.any(overlap):
        stride = num_predictions + 1
        pair_codes = (
            target_labeled[overlap].astype(np.int64, copy=False) * stride
            + pred_labeled[overlap].astype(np.int64, copy=False)
        )
        codes, intersections = np.unique(pair_codes, return_counts=True)
        target_ids = codes // stride
        pred_ids = codes % stride
        for target_id, pred_id, intersection in zip(
            target_ids, pred_ids, intersections
        ):
            if intersection > best_overlap_for_target[target_id]:
                best_overlap_for_target[target_id] = intersection
                best_pred_for_target[target_id] = pred_id
            if intersection > best_overlap_for_pred[pred_id]:
                best_overlap_for_pred[pred_id] = intersection
                best_target_for_pred[pred_id] = target_id

    target_dice = (
        2.0 * best_overlap_for_target[1:]
        / (target_areas[1:] + pred_areas[best_pred_for_target[1:]])
    )
    pred_dice = (
        2.0 * best_overlap_for_pred[1:]
        / (pred_areas[1:] + target_areas[best_target_for_pred[1:]])
    )
    target_weighted = (
        np.sum(target_areas[1:] * target_dice) / np.sum(target_areas[1:])
    )
    pred_weighted = (
        np.sum(pred_areas[1:] * pred_dice) / np.sum(pred_areas[1:])
    )
    return float(0.5 * (target_weighted + pred_weighted))


def macro_binary_metrics_from_counts(
    counts: list[tuple[int, int, int, int]],
    object_dice_values: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-image confusion counts with macro selection semantics.

    Returned ``gland_dice``/``foreground_dice``, ``background_dice`` and
    ``mdice`` are per-image macro means.  ``micro_*`` values are supplemental
    diagnostics derived after pooling pixels across images.
    """

    if not counts:
        raise RuntimeError("Validation contains no prediction/target samples")

    gland_values: list[float] = []
    background_values: list[float] = []
    for tp, fp, fn, tn in counts:
        gland_values.append(_dice_from_counts(tp, fp, fn))
        background_values.append(_dice_from_counts(tn, fn, fp))

    gland_dice = float(np.mean(gland_values))
    background_dice = float(np.mean(background_values))
    mdice = 0.5 * (gland_dice + background_dice)

    totals = np.asarray(counts, dtype=np.int64).sum(axis=0)
    total_tp, total_fp, total_fn, total_tn = map(int, totals)
    micro_gland_dice = _dice_from_counts(total_tp, total_fp, total_fn)
    micro_background_dice = _dice_from_counts(
        total_tn, total_fn, total_fp
    )

    metrics: dict[str, Any] = {
        "gland_dice": gland_dice,
        # Backward-compatible alias used by existing summaries/checkpoints.
        "foreground_dice": gland_dice,
        "background_dice": background_dice,
        "mdice": mdice,
        "samples": float(len(counts)),
        "metric_resolution": "original",
        "aggregation": "per_image_macro",
        "selection_primary_metric": "mdice",
        "selection_primary_tolerance": MDICE_CANDIDATE_TOLERANCE,
        "selection_metric": "object_dice_within_mdice_tolerance",
        "micro_gland_dice": micro_gland_dice,
        "micro_foreground_dice": micro_gland_dice,
        "micro_background_dice": micro_background_dice,
        "micro_mdice": 0.5 * (
            micro_gland_dice + micro_background_dice
        ),
    }
    if object_dice_values is not None:
        if len(object_dice_values) != len(counts):
            raise ValueError(
                "Object-Dice values and confusion counts must have equal "
                f"lengths, got {len(object_dice_values)} and {len(counts)}"
            )
        metrics["object_dice"] = float(np.mean(object_dice_values))
    return metrics


def macro_binary_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict[str, Any]:
    """Return original-resolution per-image macro binary Dice metrics."""

    if len(predictions) != len(targets):
        raise ValueError(
            "Validation predictions and targets must have equal lengths, got "
            f"{len(predictions)} and {len(targets)}"
        )
    counts = [
        binary_confusion_counts(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]
    object_dice_values = [
        calculate_object_dice(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]
    return macro_binary_metrics_from_counts(counts, object_dice_values)


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    """Return the mapping/policy key; checkpoint selection uses EarlyStopping."""

    gland_dice = metrics.get("gland_dice", metrics.get("foreground_dice"))
    if gland_dice is None or "mdice" not in metrics:
        raise KeyError("Validation metrics require mdice and gland_dice")
    return float(metrics["mdice"]), float(gland_dice)


def select_structure_aware_candidate(
    candidates: list[dict[str, Any]],
    tolerance: float = MDICE_CANDIDATE_TOLERANCE,
) -> dict[str, Any]:
    """Select max Object Dice among candidates within ``tolerance`` of mDice best."""

    if not candidates:
        raise ValueError("Structure-aware selection requires at least one candidate")
    for candidate in candidates:
        if "mdice" not in candidate or "object_dice" not in candidate:
            raise KeyError("Each candidate requires mdice and object_dice")
    best_mdice = max(float(candidate["mdice"]) for candidate in candidates)
    threshold = best_mdice - float(tolerance)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["mdice"]) >= threshold
    ]
    return max(
        eligible,
        key=lambda candidate: (
            float(candidate["object_dice"]),
            float(candidate["mdice"]),
        ),
    )


def network_metric_shape(
    original_shape: tuple[int, int] | list[int],
    input_size: int,
    *,
    preserve_aspect: bool = False,
    alignment: int = 1,
    allow_upscale: bool = True,
) -> tuple[int, int]:
    """Return the spatial shape at which a model actually sees an image.

    Square-input models always return ``input_size x input_size``.  Models that
    preserve aspect ratio use ``input_size`` as their maximum side and may align
    both dimensions to a patch size.  SGSCN can set ``allow_upscale=False`` to
    match its released per-image optimisation preprocessing.
    """
    height, width = (int(original_shape[0]), int(original_shape[1]))
    input_size = int(input_size)
    alignment = int(alignment)
    if height <= 0 or width <= 0 or input_size <= 0 or alignment <= 0:
        raise ValueError(
            "Spatial dimensions, input_size and alignment must be positive"
        )
    if not preserve_aspect:
        return input_size, input_size
    if not allow_upscale and max(height, width) <= input_size:
        return height, width
    scale = input_size / max(height, width)
    target_height = max(
        alignment,
        round((height * scale) / alignment) * alignment,
    )
    target_width = max(
        alignment,
        round((width * scale) / alignment) * alignment,
    )
    return int(target_height), int(target_width)


def resize_label_for_metrics(
    labels: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour resize for binary or integer segmentation labels."""
    labels = np.asarray(labels)
    target_shape = (int(shape[0]), int(shape[1]))
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got shape {labels.shape}")
    if labels.shape == target_shape:
        return np.array(labels, copy=True)
    tensor = torch.from_numpy(
        np.ascontiguousarray(labels.astype(np.float32, copy=False))
    )[None, None]
    resized = F.interpolate(tensor, size=target_shape, mode="nearest")[0, 0]
    if labels.dtype == np.bool_:
        return resized.numpy() >= 0.5
    return resized.round().numpy().astype(labels.dtype, copy=False)


def resize_binary_for_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a prediction/target pair to the declared metric resolution."""
    prediction = np.asarray(prediction).astype(bool, copy=False)
    target = np.asarray(target).astype(bool, copy=False)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Metric prediction and target must both be two-dimensional"
        )
    return (
        resize_label_for_metrics(prediction, shape).astype(bool, copy=False),
        resize_label_for_metrics(target, shape).astype(bool, copy=False),
    )


def format_metric_rows_for_csv(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return display rows with floating metrics rounded to two decimals."""
    allowed = {
        "sample",
        "pixel_accuracy",
        "background_iou",
        "gland_iou",
        "mean_iou",
        "background_dice",
        "gland_dice",
        "mean_dice",
        "object_dice",
    }
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        formatted: dict[str, Any] = {}
        for key, value in row.items():
            if key not in allowed:
                continue
            if isinstance(value, bool):
                formatted[key] = value
            elif isinstance(value, Integral):
                formatted[key] = int(value)
            elif isinstance(value, Real):
                formatted[key] = f"{float(value):.2f}"
            else:
                formatted[key] = value
        formatted_rows.append(formatted)
    return formatted_rows


def _annotation_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required fixed-split file is missing: {path}. "
            "Run prepare_fixed_splits.py once before training."
        )
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names:
        raise RuntimeError(f"Annotation list is empty: {path}")
    return names


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_fixed_split(annotation_dir: str | Path) -> dict[str, Any]:
    """Reject missing, reshuffled, overlapping or modified persistent splits."""
    annotation_dir = Path(annotation_dir)
    metadata_path = annotation_dir / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Fixed split metadata is missing: {metadata_path}. "
            "Run prepare_fixed_splits.py once before training."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("split_seed", -1)) != FIXED_SPLIT_SEED:
        raise RuntimeError(f"Unexpected fixed split seed in {metadata_path}")
    dataset_name = str(metadata.get("dataset", "")).lower()
    split_mode = str(metadata.get("split_mode", "")).lower()
    all_samples = split_mode == "all_samples" or dataset_name == "ebhi-seg/adenocarcinoma"
    expected_fractions = (
        (
            ("train_fraction", FIXED_TRAIN_FRACTION),
            ("validation_fraction", FIXED_VALIDATION_FRACTION),
            ("test_fraction", FIXED_TEST_FRACTION),
        )
        if all_samples
        else (
            ("train_fraction", TRAIN_VAL_ONLY_TRAIN_FRACTION),
            ("validation_fraction", TRAIN_VAL_ONLY_VALIDATION_FRACTION),
        )
    )
    for key, expected in expected_fractions:
        if float(metadata.get(key, -1.0)) != expected:
            raise RuntimeError(f"Unexpected {key} in {metadata_path}")
    if not all_samples and metadata.get("test_fraction") not in (None, "none"):
        raise RuntimeError(
            f"Train/validation-only split must not define a test fraction in {metadata_path}"
        )

    full_path = annotation_dir / "train_full.txt"
    train_path = annotation_dir / "train.txt"
    val_path = annotation_dir / "val.txt"
    test_path = annotation_dir / "test.txt"
    full = _annotation_names(full_path)
    train = _annotation_names(train_path)
    val = _annotation_names(val_path)
    test = _annotation_names(test_path)
    groups = {"train": train, "val": val, "test": test}
    for left_name, left in groups.items():
        for right_name, right in groups.items():
            if left_name < right_name and set(left).intersection(right):
                raise RuntimeError(
                    f"{left_name}/{right_name} overlap detected in {annotation_dir}"
                )
    if all_samples:
        if set(train).union(val, test) != set(full):
            raise RuntimeError(
                f"train.txt + val.txt + test.txt do not reconstruct train_full.txt in {annotation_dir}"
            )
        expected_test = int(len(full) * FIXED_TEST_FRACTION + 0.5)
        expected_val = int(len(full) * FIXED_VALIDATION_FRACTION + 0.5)
        expected_train = len(full) - expected_val - expected_test
        expected_counts = (expected_train, expected_val, expected_test)
        count_error = "7:1:2"
    else:
        if set(train).union(val) != set(full):
            raise RuntimeError(
                f"train.txt + val.txt do not reconstruct train_full.txt in {annotation_dir}"
            )
        if set(test).intersection(full):
            raise RuntimeError(
                f"Independent test list overlaps train_full.txt in {annotation_dir}"
            )
        expected_train = int(len(full) * TRAIN_VAL_ONLY_TRAIN_FRACTION + 0.5)
        expected_val = len(full) - expected_train
        expected_counts = (expected_train, expected_val, len(test))
        count_error = "8:2 train/validation"
    if (len(train), len(val), len(test)) != expected_counts:
        raise RuntimeError(
            f"Unexpected {count_error} counts in {annotation_dir}: "
            f"got {(len(train), len(val), len(test))}, expected {expected_counts}"
        )

    expected_hashes = {
        "train_full_sha256": full_path,
        "train_sha256": train_path,
        "val_sha256": val_path,
        "test_sha256_after_split": test_path,
    }
    for key, path in expected_hashes.items():
        if metadata.get(key) != _file_sha256(path):
            raise RuntimeError(f"Fixed split integrity check failed for {path}")
    return metadata


def prediction_at_original_resolution(
    output: Any,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> torch.Tensor:
    """Bilinearly restore continuous output before making a decision.

    This is the shared Fully supervised/Unsupervised protocol for continuous
    predictions. One-channel output is interpreted as probability and
    thresholded only after restoration. Two-channel output is interpreted as
    class logits and converted with argmax only after restoration. Already
    discrete class/cluster maps must instead use ``resize_label_for_metrics``,
    which performs nearest-neighbour restoration.
    """

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    if output.ndim == 3:
        output = output[:, None]
    if output.ndim != 4:
        raise ValueError(
            f"Expected continuous output with 3 or 4 dimensions, got {output.shape}"
        )

    height, width = (int(original_size[0]), int(original_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid original_size: {original_size}")

    continuous = output.float()
    if tuple(int(value) for value in continuous.shape[-2:]) != (height, width):
        continuous = F.interpolate(
            continuous,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    channels = int(continuous.shape[1])
    if channels == 1:
        return continuous[:, 0] >= float(threshold)
    if channels == 2:
        return torch.argmax(continuous, dim=1).bool()
    raise ValueError(
        "Unified binary segmentation expects one probability channel or two "
        f"logit channels, but received {channels} channels"
    )


@torch.inference_mode()
def calculate_validation_mdice(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Return original-GT-resolution per-image macro binary Dice.

    The validation dataset must expose ``original_mask`` and the loader must
    use batch size one because original image sizes may differ by sample.
    """
    was_training = model.training
    model.eval()
    predictions_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if int(images.shape[0]) != 1:
            raise ValueError(
                "Original-resolution validation requires batch_size=1, "
                f"but received {int(images.shape[0])} samples"
            )
        if "original_mask" not in batch:
            raise KeyError(
                "Validation batch is missing 'original_mask'. Construct the "
                "validation dataset with return_original=True."
            )
        targets = batch["original_mask"]
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim == 2:
            targets = targets.unsqueeze(0)
        if targets.ndim != 3 or int(targets.shape[0]) != 1:
            raise ValueError(
                f"Expected original_mask shape [1,H,W], got {tuple(targets.shape)}"
            )
        targets = targets.to(device, non_blocking=True) >= 0.5
        predictions = prediction_at_original_resolution(
            model(images),
            tuple(int(value) for value in targets.shape[-2:]),
        )

        predictions_list.append(predictions[0].detach().cpu().numpy())
        targets_list.append(targets[0].detach().cpu().numpy())

    if was_training:
        model.train()
    return macro_binary_metrics(predictions_list, targets_list)


@dataclass
class EarlyStopping:
    """Select maximum Object Dice inside the exact 0.5-pp mDice window.

    All observed validation candidates are retained as metrics, so moving the
    Mean-Dice window can select an earlier candidate that was not the winner
    when first observed. Model-state retention is handled by the caller.
    """

    patience: int = 20
    min_delta: float = 0.0001
    best: float = float("-inf")
    best_secondary: float = float("-inf")
    best_primary_observed: float = float("-inf")
    primary_tolerance: float = MDICE_CANDIDATE_TOLERANCE
    epochs_without_improvement: int = 0
    last_update_was_forced_by_primary_window: bool = False
    candidates: list[dict[str, float | int]] | None = None
    selected_index: int | None = None

    def update(self, value: float, secondary: float | None = None) -> bool:
        value = float(value)
        if secondary is None:
            raise ValueError(
                "Structure-aware checkpoint selection requires Object Dice"
            )
        secondary_value = float(secondary)
        if self.candidates is None:
            self.candidates = []
        previous_index = self.selected_index
        previous = (
            None
            if previous_index is None
            else self.candidates[previous_index]
        )
        candidate_index = len(self.candidates)
        self.candidates.append(
            {
                "candidate_index": candidate_index,
                "mdice": value,
                "object_dice": secondary_value,
            }
        )
        self.best_primary_observed = max(self.best_primary_observed, value)
        selected = select_structure_aware_candidate(
            self.candidates, self.primary_tolerance
        )
        self.selected_index = int(selected["candidate_index"])
        self.best = float(selected["mdice"])
        self.best_secondary = float(selected["object_dice"])
        threshold = self.best_primary_observed - self.primary_tolerance
        self.last_update_was_forced_by_primary_window = bool(
            previous is not None and float(previous["mdice"]) < threshold
        )
        changed = self.selected_index != previous_index
        if changed:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return changed

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience
