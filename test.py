"""Unified evaluation for the retained HP component pipelines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

ROOT = Path(__file__).resolve().parent
dataset_name = "glas"  # "adenocarcinoma" | "glas" | "pglandseg" | "all"
pipelines = "all"
device = "cuda"  # "cuda" | "cpu" | "auto"
data_root = r"D:\1.KAN\data"

_ALL_DATASETS = ["adenocarcinoma", "glas", "pglandseg"]
_DATASET_INPUT_SIZES = {
    "adenocarcinoma": 224,
    "glas": 224,
    "pglandseg": 448,
}
_PIPELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "2023HP": {
        "title": "2023 HP (hidden-positive pixel contrast)",
        "script": ROOT / "pipeline.py",
        "amp": True,
        "input_sizes": {
            "adenocarcinoma": 224,
            "glas": 224,
            "pglandseg": 448,
        },
        "test_defaults": {},
        "profile_role": "main_comparison",
    }
}
_PIPELINE_CONFIGS["HP-SGC"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + sparse region-graph consistency",
    "test_defaults": {"variant": "sgc"},
    "profile_role": "single_module_ablation",
}
_PIPELINE_CONFIGS["HP-MRFA12"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + two-radius multi-receptive-field adapter",
    "test_defaults": {"variant": "mrfa12"},
    "profile_role": "single_module_ablation",
}
_PIPELINE_CONFIGS["HP-MRFA12-SGC"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + two-radius MRFA + sparse region-graph consistency",
    "test_defaults": {"variant": "mrfa12_sgc"},
    "profile_role": "single_hyperparameter_ablation",
}
_PIPELINE_CONFIGS["HP-MRFA1-SGC"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + dilation-1 receptive-field adapter + SGC",
    "test_defaults": {"variant": "mrfa1_sgc"},
    "profile_role": "single_branch_ablation",
}
_PIPELINE_CONFIGS["HP-MRFA2-SGC"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + dilation-2 receptive-field adapter + SGC",
    "test_defaults": {"variant": "mrfa2_sgc"},
    "profile_role": "single_branch_ablation",
}
_PIPELINE_CONFIGS["HP-MRFA12-SGC-ATTR"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + two-radius MRFA + attraction-only SGC",
    "test_defaults": {"variant": "mrfa12_sgc_attr"},
    "profile_role": "single_loss_component_ablation",
}
_PIPELINE_CONFIGS["HP-MRFA12-SGC-REP"] = {
    **_PIPELINE_CONFIGS["2023HP"],
    "title": "HP + two-radius MRFA + repulsion-only SGC",
    "test_defaults": {"variant": "mrfa12_sgc_rep"},
    "profile_role": "single_loss_component_ablation",
}
_ALL_PIPELINES = list(_PIPELINE_CONFIGS)
_PIPELINE_ALIASES = {
    "2023hp": "2023HP",
    "hp": "2023HP",
    "hp-sgc": "HP-SGC",
    "sgc": "HP-SGC",
    "hp_sgc": "HP-SGC",
    "hp-mrfa12": "HP-MRFA12",
    "mrfa12": "HP-MRFA12",
    "hp_mrfa12": "HP-MRFA12",
    "hp-mrfa12-sgc": "HP-MRFA12-SGC",
    "mrfa12-sgc": "HP-MRFA12-SGC",
    "hp_mrfa12_sgc": "HP-MRFA12-SGC",
    "mrfa12_sgc": "HP-MRFA12-SGC",
    "hp-mrfa1-sgc": "HP-MRFA1-SGC",
    "mrfa1-sgc": "HP-MRFA1-SGC",
    "hp_mrfa1_sgc": "HP-MRFA1-SGC",
    "mrfa1_sgc": "HP-MRFA1-SGC",
    "hp-mrfa2-sgc": "HP-MRFA2-SGC",
    "mrfa2-sgc": "HP-MRFA2-SGC",
    "hp_mrfa2_sgc": "HP-MRFA2-SGC",
    "mrfa2_sgc": "HP-MRFA2-SGC",
    "hp-mrfa12-sgc-attr": "HP-MRFA12-SGC-ATTR",
    "mrfa12-sgc-attr": "HP-MRFA12-SGC-ATTR",
    "hp_mrfa12_sgc_attr": "HP-MRFA12-SGC-ATTR",
    "mrfa12_sgc_attr": "HP-MRFA12-SGC-ATTR",
    "hp-mrfa12-sgc-rep": "HP-MRFA12-SGC-REP",
    "mrfa12-sgc-rep": "HP-MRFA12-SGC-REP",
    "hp_mrfa12_sgc_rep": "HP-MRFA12-SGC-REP",
    "mrfa12_sgc_rep": "HP-MRFA12-SGC-REP",
    "best": "HP-MRFA12-SGC",
}


def _options_as_argv(options: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for name, value in options.items():
        if value is None or value is False:
            continue
        option = f"--{name}"
        if value is True:
            argv.append(option)
        elif isinstance(value, (tuple, list)):
            argv.extend([option, *(str(item) for item in value)])
        else:
            argv.extend([option, str(value)])
    return argv


def _run_pipeline(config: dict[str, Any], argv: list[str]) -> None:
    script = Path(config["script"])
    if not script.is_file():
        raise FileNotFoundError(f"Pipeline adapter not found: {script}")
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.run([sys.executable, str(script), *argv], cwd=str(ROOT), check=True)


def _last_option_value(argv: list[str], option: str) -> str | None:
    indices = [index for index, value in enumerate(argv[:-1]) if value == option]
    return argv[indices[-1] + 1] if indices else None


def _select_datasets(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(_ALL_DATASETS)
    if value not in _ALL_DATASETS:
        raise ValueError(f"Unknown dataset {value!r}; choose adenocarcinoma | glas | pglandseg | all")
    return [value]


def _select_pipelines(value: str) -> list[str]:
    value = value.strip()
    if value.lower() == "all":
        return list(_ALL_PIPELINES)
    canonical = _PIPELINE_ALIASES.get(value.lower(), value)
    if canonical not in _PIPELINE_CONFIGS:
        choices = " | ".join(_ALL_PIPELINES)
        raise ValueError(f"Unknown pipeline {value!r}; choose {choices} | all")
    return [canonical]


def _spatial_arguments(pipeline: str, input_size: int | None = None) -> list[str]:
    # ``pipeline`` is retained as the first argument for callers of the old
    # two-argument helper; every current variant uses the same HP geometry.
    if input_size is None:
        input_size = int(pipeline)
    return ["--input_size", str(input_size)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test HP baseline, component, or combination pipelines",
        epilog="Model-specific arguments are forwarded to pipeline.py, for example: --max_samples 1",
    )
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument("--pipeline", default=pipelines)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--dino_checkpoint", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser


def main() -> None:
    args, extra_args = _build_parser().parse_known_args()
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    selected_datasets = _select_datasets(args.dataset)
    selected_pipelines = _select_pipelines(args.pipeline)
    tasks = [(pipeline, dataset) for pipeline in selected_pipelines for dataset in selected_datasets]
    if args.checkpoint is not None and len(tasks) != 1:
        raise ValueError("--checkpoint requires exactly one pipeline and one dataset")

    print(f"\n[test_unified] {len(tasks)} task(s): {selected_pipelines} on {selected_datasets}\n")
    completed = 0
    for index, (pipeline, dataset) in enumerate(tasks, start=1):
        config = _PIPELINE_CONFIGS[pipeline]
        checkpoint = (
            Path(args.checkpoint)
            if args.checkpoint
            else ROOT / "train_out" / pipeline / dataset / "checkpoint_final.pth"
        )
        if not checkpoint.is_file():
            print(f"[SKIP] Missing checkpoint: {checkpoint}\n       Run train.py first.")
            continue
        input_size = int(config["input_sizes"][dataset])
        output_dir = ROOT / "test_out" / pipeline / dataset
        argv = [
            "--mode", "test",
            "--dataset", dataset,
            "--data_root", args.data_root or data_root,
            "--device", args.device or device,
            "--checkpoint", str(checkpoint),
            "--output_dir", str(output_dir),
            "--split", args.split,
            "--visual_max_side", "512",
            *_spatial_arguments(pipeline, input_size),
        ]
        if args.dino_checkpoint:
            argv.extend(["--dino_checkpoint", str(args.dino_checkpoint)])
        if config["amp"]:
            argv.append("--amp")
        argv.extend(_options_as_argv(config.get("test_defaults", {})))
        argv.extend([*extra_args, "--metric_resolution", "original"])
        expected_variant = str(
            config.get("test_defaults", {}).get("variant", "none")
        )
        actual_variant = _last_option_value(argv, "--variant") or "none"
        if actual_variant != expected_variant:
            raise ValueError(
                f"Pipeline {pipeline} requires --variant={expected_variant}; "
                f"got {actual_variant}"
            )

        print(f"\n{'=' * 72}\n  [{index}/{len(tasks)}] {config['title']} on {dataset.upper()} {args.split.upper()}\n{'=' * 72}\n")
        _run_pipeline(config, argv)

        actual_output = Path(_last_option_value(argv, "--output_dir") or output_dir)
        actual_output.mkdir(parents=True, exist_ok=True)
        protocol_path = actual_output / "evaluation_protocol.json"
        protocol: dict[str, Any] = {}
        if protocol_path.is_file():
            try:
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                protocol = {}
        protocol.update(
            {
                "metric_resolution": "original",
                "checkpoint": str(checkpoint),
                "dataset": dataset,
                "pipeline": pipeline,
                "profile_role": config.get("profile_role", "main_comparison"),
                "split": args.split,
                "input_size": input_size,
                "object_dice_definition": (
                    "classic_glas_bidirectional_area_weighted_8_connected"
                ),
            }
        )
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        completed += 1

    print(f"\n[test_unified] Finished {completed}/{len(tasks)} evaluation(s).")


if __name__ == "__main__":
    main()
