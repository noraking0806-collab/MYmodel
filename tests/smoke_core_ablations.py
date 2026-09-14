"""Tiny train/reload/evaluate smoke run for the four core ablations."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import main  # noqa: E402


DATA_ROOT = Path(r"D:\1.KAN\data")
CHECKPOINT = ROOT / "dino_deitsmall8_300ep_pretrain.pth"
PROFILES = {
    "mrfa1_sgc": ["--context_dilations", "1"],
    "mrfa2_sgc": ["--context_dilations", "2"],
    "mrfa12_sgc_attr": [
        "--context_dilations",
        "1",
        "2",
        "--graph_repulsion_weight",
        "0",
    ],
    "mrfa12_sgc_rep": [
        "--context_dilations",
        "1",
        "2",
        "--graph_attraction_weight",
        "0",
    ],
}


def run() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="hp_core_ablation_smoke_") as temp:
        temp_root = Path(temp)
        for variant, component_arguments in PROFILES.items():
            train_output = temp_root / variant / "train"
            main(
                [
                    "--mode",
                    "train",
                    "--dataset",
                    "glas",
                    "--data_root",
                    str(DATA_ROOT),
                    "--device",
                    device,
                    "--input_size",
                    "32",
                    "--patch_size",
                    "8",
                    "--batch_size",
                    "2",
                    "--num_workers",
                    "0",
                    "--epochs",
                    "1",
                    "--dim",
                    "32",
                    "--pool_size",
                    "32",
                    "--renew_interval",
                    "0",
                    "--warmup_steps",
                    "0",
                    "--patience",
                    "1",
                    "--max_samples",
                    "4",
                    "--max_val_samples",
                    "1",
                    "--variant",
                    variant,
                    "--dino_checkpoint",
                    str(CHECKPOINT),
                    "--no_dino_download",
                    "--output_dir",
                    str(train_output),
                    *component_arguments,
                ]
            )
            checkpoint = train_output / "checkpoint_final.pth"
            if not checkpoint.is_file():
                raise AssertionError(f"Missing smoke checkpoint for {variant}")

            test_output = temp_root / variant / "test"
            main(
                [
                    "--mode",
                    "test",
                    "--dataset",
                    "glas",
                    "--data_root",
                    str(DATA_ROOT),
                    "--device",
                    device,
                    "--input_size",
                    "32",
                    "--patch_size",
                    "8",
                    "--variant",
                    variant,
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "val",
                    "--max_samples",
                    "1",
                    "--output_dir",
                    str(test_output),
                ]
            )
            if not (test_output / "summary.txt").is_file():
                raise AssertionError(f"Missing smoke summary for {variant}")
            print(f"[smoke] {variant}: train, checkpoint reload, and test passed")


if __name__ == "__main__":
    run()
