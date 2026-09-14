"""Paired, reproducible comparison of HP pipeline metric CSV files.

The evaluator writes per-image values rounded to two decimal places.  This
utility deliberately reports that limitation and never reads images or masks.
It is intended for post-hoc diagnosis, not checkpoint or hyperparameter
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ("mean_iou", "gland_dice", "mean_dice", "object_dice")


def _read_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["sample"]: {metric: float(row[metric]) for metric in METRICS}
        for row in rows
        if row["sample"] != "MEAN"
    }


def _bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(samples, len(differences)),
    )
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compare(
    candidate_path: Path,
    reference_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    candidate = _read_metrics(candidate_path)
    reference = _read_metrics(reference_path)
    if set(candidate) != set(reference):
        raise ValueError("Candidate and reference sample sets do not match")
    names = sorted(candidate)
    output: dict[str, object] = {
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "sample_count": len(names),
        "csv_precision_warning": (
            "Paired statistics use evaluator CSV values rounded to two decimals"
        ),
        "metrics": {},
        "subsets": {},
    }
    metric_output = output["metrics"]
    assert isinstance(metric_output, dict)
    for metric_index, metric in enumerate(METRICS):
        differences = np.asarray(
            [candidate[name][metric] - reference[name][metric] for name in names]
        )
        low, high = _bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=seed + metric_index,
        )
        nonzero = differences[differences != 0.0]
        p_value = (
            float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            if len(nonzero)
            else 1.0
        )
        metric_output[metric] = {
            "candidate_mean_percent": 100.0
            * float(np.mean([candidate[name][metric] for name in names])),
            "reference_mean_percent": 100.0
            * float(np.mean([reference[name][metric] for name in names])),
            "paired_mean_change_pp": 100.0 * float(differences.mean()),
            "bootstrap_95_ci_pp": [100.0 * low, 100.0 * high],
            "wilcoxon_p": p_value,
            "wins_losses_ties": [
                int((differences > 0.0).sum()),
                int((differences < 0.0).sum()),
                int((differences == 0.0).sum()),
            ],
        }

    subsets = output["subsets"]
    assert isinstance(subsets, dict)
    for subset, prefix in (("test_a", "testA_"), ("test_b", "testB_")):
        selected = [name for name in names if name.startswith(prefix)]
        subset_metrics = {
            metric: {
                "candidate_mean_percent": 100.0
                * float(np.mean([candidate[name][metric] for name in selected])),
                "reference_mean_percent": 100.0
                * float(np.mean([reference[name][metric] for name in selected])),
            }
            for metric in METRICS
        }
        subsets[subset] = {
            "sample_count": len(selected),
            **subset_metrics,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.candidate,
                args.reference,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
