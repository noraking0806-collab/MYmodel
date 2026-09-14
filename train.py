"""Unified training entry point for the retained HP component pipelines.

The actual model, hidden-positive objective and training loop are the
source-aligned implementation in :mod:`pipeline`.  This wrapper only
resolves the fixed comparison profile and launches that adapter in a clean
Python process.
"""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_DINO_SMALL8_CHECKPOINT = ROOT / "dino_deitsmall8_300ep_pretrain.pth"

# User-facing defaults.  Dataset-specific geometry and optimisation values
# follow the comparison profile used by 2023HP-main.
dataset_name = "glas"  # "adenocarcinoma" | "glas" | "pglandseg" | "all"
pipelines = "all"
device = "cuda"  # "cuda" | "cpu" | "auto"
data_root = r"D:\1.KAN\data"

_ALL_DATASETS = ["adenocarcinoma", "glas", "pglandseg"]
_DATASET_PROFILE_CONTEXT = {
    "adenocarcinoma": {"train_images": 556, "native_geometry": "224x224"},
    "glas": {"train_images": 68, "native_geometry": "567-775 wide x 430-522 high"},
    "pglandseg": {"train_images": 800, "native_geometry": "1500x1500"},
}
_PARAMETER_POLICY = (
    "2023HP retains the original HP objective and DINO-S/8 backbone; "
    "Adenocarcinoma/Glas use 224 and PGlandSeg uses 448"
)

_PIPELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "2023HP": {
        "title": "2023 HP (hidden-positive pixel contrast)",
        "script": ROOT / "pipeline.py",
        "profile_role": "main_comparison",
        "parameter_policy": _PARAMETER_POLICY,
        "defaults": {
            "patch_size": 8,
            "dim": 512,
            "temperature": 0.8,
            "alpha": 0.05,
            "rho": 0.02,
            "ema_m": 0.99,
            "weight_decay": 0.1,
            "seed": 42,
            "amp": True,
        },
        "datasets": {
            "adenocarcinoma": {
                "input_size": 224,
                "batch_size": 8,
                "epochs": 80,
                "pool_size": 2048,
                "renew_interval": 100,
                "learning_rate": 0.0005,
                "cluster_learning_rate": 0.005,
                "patience": 20,
            },
            "glas": {
                "input_size": 224,
                "batch_size": 8,
                "epochs": 180,
                "pool_size": 2048,
                "renew_interval": 100,
                "learning_rate": 0.0005,
                "cluster_learning_rate": 0.005,
                "patience": 30,
            },
            "pglandseg": {
                "input_size": 448,
                "batch_size": 2,
                "epochs": 60,
                "pool_size": 2048,
                "renew_interval": 100,
                "learning_rate": 0.0005,
                "cluster_learning_rate": 0.005,
                "patience": 10,
            },
        },
        "weights": [
            {
                "option": "dino_checkpoint",
                "override": "dino_checkpoint",
                "default": DEFAULT_DINO_SMALL8_CHECKPOINT,
                "error": "2023HP requires DINO-S/8 pretrained weights",
            }
        ],
    }
}

sgc_config = copy.deepcopy(_PIPELINE_CONFIGS["2023HP"])
sgc_config["title"] = "HP + sparse region-graph consistency"
sgc_config["profile_role"] = "single_module_ablation"
sgc_config["defaults"].update(
    {
        "variant": "sgc",
        "graph_weight": 0.10,
        "graph_grid_size": 6,
        "graph_neighbors": 4,
        "graph_attraction_weight": 1.0,
        "graph_repulsion_weight": 0.50,
        "graph_negative_quantile": 0.25,
        "graph_prediction_temperature": 0.20,
        "graph_negative_margin": 0.25,
    }
)
sgc_config["datasets"]["pglandseg"]["graph_grid_size"] = 14
sgc_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; add only sparse region-graph consistency"
)
sgc_config["weights"][0]["error"] = (
    "HP-SGC requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-SGC"] = sgc_config

mrfa12_config = copy.deepcopy(_PIPELINE_CONFIGS["2023HP"])
mrfa12_config["title"] = "HP + two-radius multi-receptive-field adapter"
mrfa12_config["profile_role"] = "single_module_ablation"
mrfa12_config["defaults"].update(
    {
        "variant": "mrfa12",
        "context_bottleneck": 64,
        "context_dilations": [1, 2],
    }
)
mrfa12_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; add only the two-radius multi-receptive-field "
    "adapter with dilations {1,2}"
)
mrfa12_config["weights"][0]["error"] = (
    "HP-MRFA12 requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-MRFA12"] = mrfa12_config

mrfa12_sgc_config = copy.deepcopy(mrfa12_config)
mrfa12_sgc_config["title"] = (
    "HP + two-radius MRFA + sparse region-graph consistency"
)
mrfa12_sgc_config["profile_role"] = "combined_component_interaction"
mrfa12_sgc_config["defaults"].update(
    {
        "variant": "mrfa12_sgc",
        "graph_weight": 0.10,
        "graph_grid_size": 6,
        "graph_neighbors": 4,
        "graph_attraction_weight": 1.0,
        "graph_repulsion_weight": 0.50,
        "graph_negative_quantile": 0.25,
        "graph_prediction_temperature": 0.20,
        "graph_negative_margin": 0.25,
    }
)
mrfa12_sgc_config["datasets"]["pglandseg"]["graph_grid_size"] = 14
mrfa12_sgc_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; combine the two-radius MRFA adapter and SGC loss"
)
mrfa12_sgc_config["weights"][0]["error"] = (
    "HP-MRFA12-SGC requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-MRFA12-SGC"] = mrfa12_sgc_config

mrfa1_sgc_config = copy.deepcopy(mrfa12_sgc_config)
mrfa1_sgc_config["title"] = (
    "HP + dilation-1 receptive-field adapter + sparse region-graph consistency"
)
mrfa1_sgc_config["profile_role"] = "single_branch_ablation"
mrfa1_sgc_config["defaults"].update(
    {
        "variant": "mrfa1_sgc",
        "context_dilations": [1],
    }
)
mrfa1_sgc_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; relative to HP-MRFA12-SGC change only MRFA "
    "dilations from {1,2} to {1}"
)
mrfa1_sgc_config["weights"][0]["error"] = (
    "HP-MRFA1-SGC requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-MRFA1-SGC"] = mrfa1_sgc_config

mrfa2_sgc_config = copy.deepcopy(mrfa12_sgc_config)
mrfa2_sgc_config["title"] = (
    "HP + dilation-2 receptive-field adapter + sparse region-graph consistency"
)
mrfa2_sgc_config["profile_role"] = "single_branch_ablation"
mrfa2_sgc_config["defaults"].update(
    {
        "variant": "mrfa2_sgc",
        "context_dilations": [2],
    }
)
mrfa2_sgc_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; relative to HP-MRFA12-SGC change only MRFA "
    "dilations from {1,2} to {2}"
)
mrfa2_sgc_config["weights"][0]["error"] = (
    "HP-MRFA2-SGC requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-MRFA2-SGC"] = mrfa2_sgc_config

mrfa12_sgc_attr_config = copy.deepcopy(mrfa12_sgc_config)
mrfa12_sgc_attr_config["title"] = (
    "HP + two-radius MRFA + attraction-only sparse region-graph consistency"
)
mrfa12_sgc_attr_config["profile_role"] = "single_loss_component_ablation"
mrfa12_sgc_attr_config["defaults"].update(
    {
        "variant": "mrfa12_sgc_attr",
        "graph_repulsion_weight": 0.0,
    }
)
mrfa12_sgc_attr_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; relative to HP-MRFA12-SGC disable only boundary "
    "repulsion by changing graph_repulsion_weight from 0.50 to 0"
)
mrfa12_sgc_attr_config["weights"][0]["error"] = (
    "HP-MRFA12-SGC-ATTR requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-MRFA12-SGC-ATTR"] = mrfa12_sgc_attr_config

mrfa12_sgc_rep_config = copy.deepcopy(mrfa12_sgc_config)
mrfa12_sgc_rep_config["title"] = (
    "HP + two-radius MRFA + repulsion-only sparse region-graph consistency"
)
mrfa12_sgc_rep_config["profile_role"] = "single_loss_component_ablation"
mrfa12_sgc_rep_config["defaults"].update(
    {
        "variant": "mrfa12_sgc_rep",
        "graph_attraction_weight": 0.0,
    }
)
mrfa12_sgc_rep_config["parameter_policy"] = (
    f"{_PARAMETER_POLICY}; relative to HP-MRFA12-SGC disable only positive-edge "
    "attraction by changing graph_attraction_weight from 1 to 0"
)
mrfa12_sgc_rep_config["weights"][0]["error"] = (
    "HP-MRFA12-SGC-REP requires DINO-S/8 pretrained weights"
)
_PIPELINE_CONFIGS["HP-MRFA12-SGC-REP"] = mrfa12_sgc_rep_config

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


def _required_weight_arguments(
    config: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    argv: list[str] = []
    for weight in config.get("weights", []):
        override_name = weight.get("override")
        override_value = getattr(args, override_name, None) if override_name else None
        path = Path(override_value) if override_value else Path(weight["default"])
        if not path.is_file() and not args.config_only:
            raise FileNotFoundError(f"{weight['error']}: {path}")
        argv.extend([f"--{weight['option']}", str(path)])
    return argv


def _last_option_value(argv: list[str], option: str) -> str | None:
    indices = [index for index, value in enumerate(argv[:-1]) if value == option]
    return argv[indices[-1] + 1] if indices else None


def _arguments_as_mapping(argv: list[str]) -> dict[str, object]:
    resolved: dict[str, object] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:]
        end = index + 1
        while end < len(argv) and not argv[end].startswith("--"):
            end += 1
        values = argv[index + 1 : end]
        resolved[key] = True if not values else values[0] if len(values) == 1 else values
        index = end
    return resolved


def _build_train_arguments(
    pipeline: str, dataset: str, args: argparse.Namespace
) -> list[str]:
    config = _PIPELINE_CONFIGS[pipeline]
    argv = [
        "--mode", "train",
        "--dataset", dataset,
        "--data_root", args.data_root or data_root,
        "--device", args.device or device,
    ]
    argv.extend(_options_as_argv(config["defaults"]))
    argv.extend(_options_as_argv(config["datasets"][dataset]))
    argv.extend(_required_weight_arguments(config, args))
    return argv


def _run_pipeline(config: dict[str, Any], argv: list[str]) -> None:
    script = Path(config["script"])
    if not script.is_file():
        raise FileNotFoundError(f"Pipeline adapter not found: {script}")
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.run([sys.executable, str(script), *argv], cwd=str(ROOT), check=True)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train HP baseline, component, or combination pipelines",
        epilog=(
            "Model-specific arguments are forwarded to pipeline.py, for example: "
            "--epochs 1 --max_samples 8 --max_val_samples 2"
        ),
    )
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument("--pipeline", default=pipelines)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--dino_checkpoint", default=None)
    parser.add_argument("--config_only", action="store_true")
    return parser


def main() -> None:
    args, extra_args = _build_parser().parse_known_args()
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    selected_datasets = _select_datasets(args.dataset)
    selected_pipelines = _select_pipelines(args.pipeline)
    tasks = [(pipeline, dataset) for pipeline in selected_pipelines for dataset in selected_datasets]

    print(f"\n[train_unified] {len(tasks)} task(s): {selected_pipelines} on {selected_datasets}\n")
    resolved_tasks: list[dict[str, object]] = []
    for index, (pipeline, dataset) in enumerate(tasks, start=1):
        config = _PIPELINE_CONFIGS[pipeline]
        output_dir = ROOT / "train_out" / pipeline / dataset
        argv = _build_train_arguments(pipeline, dataset, args)
        argv.extend(["--output_dir", str(output_dir), *extra_args])
        expected_size = int(config["datasets"][dataset]["input_size"])
        actual_size = _last_option_value(argv, "--input_size")
        if actual_size != str(expected_size):
            raise ValueError(
                f"The HP comparison locks {dataset} --input_size={expected_size}; "
                f"got {actual_size}"
            )
        expected_variant = str(config["defaults"].get("variant", "none"))
        actual_variant = _last_option_value(argv, "--variant") or "none"
        if actual_variant != expected_variant:
            raise ValueError(
                f"Pipeline {pipeline} requires --variant={expected_variant}; "
                f"got {actual_variant}"
            )
        argv.extend(["--metric_resolution", "original"])
        resolved_arguments = _arguments_as_mapping(argv)
        if args.config_only:
            resolved_tasks.append(
                {
                    "pipeline": pipeline,
                    "dataset": dataset,
                    "dataset_context": _DATASET_PROFILE_CONTEXT[dataset],
                    "parameter_policy": config.get("parameter_policy", _PARAMETER_POLICY),
                    "dataset_input_size": expected_size,
                    "adapter_arguments": resolved_arguments,
                }
            )
            continue
        print(f"\n{'=' * 72}\n  [{index}/{len(tasks)}] {config['title']} on {dataset.upper()}\n{'=' * 72}\n")
        _run_pipeline(config, argv)
        actual_output = Path(_last_option_value(argv, "--output_dir") or output_dir)
        actual_output.mkdir(parents=True, exist_ok=True)
        protocol_path = actual_output / "comparison_protocol.json"
        protocol: dict[str, object] = {}
        if protocol_path.is_file():
            try:
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                protocol = {}
        protocol.update(
            {
                "metric_resolution": "original",
                "dataset": dataset,
                "pipeline": pipeline,
                "profile_role": config.get("profile_role", "main_comparison"),
                "parameter_policy": config.get("parameter_policy", _PARAMETER_POLICY),
                "dataset_input_size": expected_size,
                "dataset_context": _DATASET_PROFILE_CONTEXT[dataset],
                "resolved_adapter_arguments": resolved_arguments,
            }
        )
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    if args.config_only:
        print(json.dumps({"training_started": False, "task_count": len(resolved_tasks), "tasks": resolved_tasks}, indent=2))
        print(f"\n[train_unified] Resolved {len(resolved_tasks)} task(s); no training was started.")
    else:
        print(f"\n[train_unified] All {len(tasks)} task(s) finished.")


if __name__ == "__main__":
    main()
