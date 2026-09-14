from __future__ import annotations

import argparse

import torch

from multi_receptive_field import MultiReceptiveFieldAdapter
import test as test_wrapper
import train as train_wrapper
from pipeline import (
    HPFeaturizer,
    PIPELINE_MODEL_NAMES,
    _capture_validation_candidate_state,
    _component_hparams,
    _restore_validation_candidate_state,
    _retained_validation_candidate_indices,
    canonical_variant_name,
    model_name_for_variant,
    variant_uses_mrfa,
    variant_uses_sgc,
)
from sparse_region_graph import SparseRegionGraphConsistency
from utils.validation import (
    EarlyStopping,
    MDICE_CANDIDATE_TOLERANCE,
    select_structure_aware_candidate,
)


RETAINED_PIPELINES = (
    "2023HP",
    "HP-SGC",
    "HP-MRFA12",
    "HP-MRFA12-SGC",
    "HP-MRFA1-SGC",
    "HP-MRFA2-SGC",
    "HP-MRFA12-SGC-ATTR",
    "HP-MRFA12-SGC-REP",
)
RETAINED_VARIANTS = {
    "none": "2023HP",
    "sgc": "HP-SGC",
    "mrfa12": "HP-MRFA12",
    "mrfa12_sgc": "HP-MRFA12-SGC",
    "mrfa1_sgc": "HP-MRFA1-SGC",
    "mrfa2_sgc": "HP-MRFA2-SGC",
    "mrfa12_sgc_attr": "HP-MRFA12-SGC-ATTR",
    "mrfa12_sgc_rep": "HP-MRFA12-SGC-REP",
}


def test_only_retained_pipelines_are_registered() -> None:
    assert PIPELINE_MODEL_NAMES == RETAINED_VARIANTS
    assert tuple(train_wrapper._PIPELINE_CONFIGS) == RETAINED_PIPELINES
    assert tuple(test_wrapper._PIPELINE_CONFIGS) == RETAINED_PIPELINES
    assert tuple(train_wrapper._ALL_PIPELINES) == RETAINED_PIPELINES
    assert tuple(test_wrapper._ALL_PIPELINES) == RETAINED_PIPELINES
    for variant, pipeline in RETAINED_VARIANTS.items():
        assert canonical_variant_name(variant) == variant
        assert model_name_for_variant(variant) == pipeline


def test_retained_variant_component_routing() -> None:
    assert not variant_uses_mrfa("none")
    assert not variant_uses_sgc("none")
    assert not variant_uses_mrfa("sgc")
    assert variant_uses_sgc("sgc")
    assert variant_uses_mrfa("mrfa12")
    assert not variant_uses_sgc("mrfa12")
    for variant in (
        "mrfa12_sgc",
        "mrfa1_sgc",
        "mrfa2_sgc",
        "mrfa12_sgc_attr",
        "mrfa12_sgc_rep",
    ):
        assert variant_uses_mrfa(variant)
        assert variant_uses_sgc(variant)


def test_retained_aliases_and_best_target() -> None:
    for wrapper in (train_wrapper, test_wrapper):
        assert wrapper._select_pipelines("hp") == ["2023HP"]
        assert wrapper._select_pipelines("sgc") == ["HP-SGC"]
        assert wrapper._select_pipelines("mrfa12") == ["HP-MRFA12"]
        assert wrapper._select_pipelines("mrfa12-sgc") == ["HP-MRFA12-SGC"]
        assert wrapper._select_pipelines("mrfa1_sgc") == ["HP-MRFA1-SGC"]
        assert wrapper._select_pipelines("mrfa2-sgc") == ["HP-MRFA2-SGC"]
        assert wrapper._select_pipelines("mrfa12_sgc_attr") == [
            "HP-MRFA12-SGC-ATTR"
        ]
        assert wrapper._select_pipelines("mrfa12-sgc-rep") == [
            "HP-MRFA12-SGC-REP"
        ]
        assert wrapper._select_pipelines("best") == ["HP-MRFA12-SGC"]
        assert set(wrapper._PIPELINE_ALIASES.values()) <= set(RETAINED_PIPELINES)


def test_train_configs_preserve_controlled_ablation_contracts() -> None:
    configs = train_wrapper._PIPELINE_CONFIGS
    baseline = configs["2023HP"]["defaults"]
    sgc = configs["HP-SGC"]["defaults"]
    mrfa12 = configs["HP-MRFA12"]["defaults"]
    full = configs["HP-MRFA12-SGC"]["defaults"]

    assert {key: value for key, value in mrfa12.items() if key in baseline} == baseline
    assert mrfa12["variant"] == "mrfa12"
    assert mrfa12["context_bottleneck"] == 64
    assert mrfa12["context_dilations"] == [1, 2]

    graph_keys = {
        "graph_weight",
        "graph_grid_size",
        "graph_neighbors",
        "graph_attraction_weight",
        "graph_repulsion_weight",
        "graph_negative_quantile",
        "graph_prediction_temperature",
        "graph_negative_margin",
    }
    assert {key: full[key] for key in graph_keys} == {
        key: sgc[key] for key in graph_keys
    }
    assert full["context_bottleneck"] == mrfa12["context_bottleneck"]
    assert full["context_dilations"] == mrfa12["context_dilations"]
    assert configs["HP-SGC"]["datasets"]["pglandseg"]["graph_grid_size"] == 14
    assert configs["HP-MRFA12-SGC"]["datasets"]["pglandseg"]["graph_grid_size"] == 14

    expected_changes = {
        "HP-MRFA1-SGC": {
            "variant": "mrfa1_sgc",
            "context_dilations": [1],
        },
        "HP-MRFA2-SGC": {
            "variant": "mrfa2_sgc",
            "context_dilations": [2],
        },
        "HP-MRFA12-SGC-ATTR": {
            "variant": "mrfa12_sgc_attr",
            "graph_repulsion_weight": 0.0,
        },
        "HP-MRFA12-SGC-REP": {
            "variant": "mrfa12_sgc_rep",
            "graph_attraction_weight": 0.0,
        },
    }
    for pipeline, changes in expected_changes.items():
        expected = dict(full)
        expected.update(changes)
        assert configs[pipeline]["defaults"] == expected


def test_all_expands_to_twenty_four_retained_tasks() -> None:
    pipelines = train_wrapper._select_pipelines("all")
    datasets = train_wrapper._select_datasets("all")
    tasks = [(pipeline, dataset) for pipeline in pipelines for dataset in datasets]
    assert len(pipelines) == 8
    assert len(datasets) == 3
    assert len(tasks) == 24
    assert len(set(tasks)) == 24


def test_sparse_region_graph_is_parameter_free_and_gradient_safe() -> None:
    torch.manual_seed(7)
    module = SparseRegionGraphConsistency(grid_size=3, neighbors=4)
    online = torch.randn(2, 8, 8, 8, requires_grad=True)
    teacher = torch.randn(2, 8, 8, 8, requires_grad=True)
    centres = torch.randn(2, 8, requires_grad=True)
    loss, statistics = module(online, teacher, centres)
    assert torch.isfinite(loss)
    assert statistics["positive_edges_per_node"] == 4.0
    assert list(module.parameters()) == []
    loss.backward()
    assert online.grad is not None
    assert torch.isfinite(online.grad).all()
    assert teacher.grad is None
    assert centres.grad is None


def test_sparse_region_graph_loss_components_are_independently_switchable() -> None:
    torch.manual_seed(31)
    online_values = torch.randn(2, 8, 8, 8)
    teacher = torch.randn(2, 8, 8, 8, requires_grad=True)
    centres = torch.randn(2, 8, requires_grad=True)

    def evaluate(attraction: float, repulsion: float) -> torch.Tensor:
        online = online_values.detach().clone().requires_grad_(True)
        module = SparseRegionGraphConsistency(
            attraction_weight=attraction,
            repulsion_weight=repulsion,
        )
        loss, _ = module(online, teacher, centres)
        loss.backward()
        assert online.grad is not None
        assert torch.isfinite(online.grad).all()
        return loss.detach()

    full = evaluate(1.0, 0.5)
    attraction_only = evaluate(1.0, 0.0)
    repulsion_only = evaluate(0.0, 0.5)
    torch.testing.assert_close(full, attraction_only + repulsion_only)
    assert teacher.grad is None
    assert centres.grad is None


def test_multi_receptive_field_adapter_is_near_identity_and_trainable() -> None:
    torch.manual_seed(11)
    module = MultiReceptiveFieldAdapter(
        channels=16, bottleneck=8, dilations=(1, 2)
    )
    features = torch.randn(2, 16, 9, 9, requires_grad=True)
    output = module(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert float((output - features).abs().mean().detach()) < 0.01
    output.square().mean().backward()
    assert features.grad is not None
    assert module.residual_gate.grad is not None
    assert any(
        parameter.grad is not None
        for name, parameter in module.named_parameters()
        if name != "residual_gate"
    )


def test_component_hparams_include_attraction_weight() -> None:
    args = argparse.Namespace(
        variant="mrfa12_sgc",
        context_bottleneck=48,
        context_dilations=[1, 2],
        graph_weight=0.07,
        graph_grid_size=5,
        graph_neighbors=3,
        graph_attraction_weight=0.6,
        graph_repulsion_weight=0.4,
        graph_negative_quantile=0.2,
        graph_prediction_temperature=0.3,
        graph_negative_margin=0.15,
    )
    assert _component_hparams(args) == {
        "graph_weight": 0.07,
        "graph_grid_size": 5,
        "graph_neighbors": 3,
        "graph_attraction_weight": 0.6,
        "graph_repulsion_weight": 0.4,
        "graph_negative_quantile": 0.2,
        "graph_prediction_temperature": 0.3,
        "graph_negative_margin": 0.15,
        "context_bottleneck": 48,
        "context_dilations": [1, 2],
    }


def _build_tiny_hp_model(variant: str) -> HPFeaturizer:
    torch.manual_seed(101)
    return HPFeaturizer(
        dim=8,
        input_size=16,
        patch_size=8,
        dropout=0.0,
        load_pretrained=False,
        allow_download=False,
        variant=variant,
    )


def test_sgc_preserves_baseline_inference_state() -> None:
    baseline = _build_tiny_hp_model("none").eval()
    sgc = _build_tiny_hp_model("sgc").eval()
    image = torch.randn(1, 3, 16, 16)
    assert sgc.context_adapter is None
    assert sgc.ema_context_adapter is None
    assert list(sgc.state_dict()) == list(baseline.state_dict())
    for key, value in sgc.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[key])
    with torch.inference_mode():
        baseline_outputs = baseline(image, hp_train=False)
        sgc_outputs = sgc(image, hp_train=False)
    for actual, expected in zip(sgc_outputs, baseline_outputs):
        torch.testing.assert_close(actual, expected)


def test_mrfa12_checkpoint_geometry_restores_strictly() -> None:
    source = HPFeaturizer(
        dim=8,
        input_size=16,
        patch_size=8,
        dropout=0.0,
        load_pretrained=False,
        allow_download=False,
        variant="mrfa12",
        context_bottleneck=12,
        context_dilations=(1, 4),
    )
    payload = {
        "variant": "mrfa12",
        "dim": 8,
        "input_size": 16,
        "patch_size": 8,
        "ema_m": 0.99,
        "component_hparams": {
            "context_bottleneck": 12,
            "context_dilations": [1, 4],
        },
        "model": source.state_dict(),
    }
    component = payload["component_hparams"]
    restored = HPFeaturizer(
        dim=int(payload["dim"]),
        input_size=int(payload["input_size"]),
        patch_size=int(payload["patch_size"]),
        ema_m=float(payload["ema_m"]),
        load_pretrained=False,
        allow_download=False,
        variant=str(payload["variant"]),
        context_bottleneck=int(component["context_bottleneck"]),
        context_dilations=tuple(component["context_dilations"]),
    )
    restored.load_state_dict(payload["model"], strict=True)
    assert restored.context_adapter is not None
    assert restored.context_adapter.bottleneck == 12
    assert restored.context_adapter.dilations == (1, 4)


def test_structure_aware_selection_uses_exact_half_pp_window() -> None:
    assert MDICE_CANDIDATE_TOLERANCE == 0.005
    candidates = [
        {"candidate_index": 0, "mdice": 0.8000, "object_dice": 0.10},
        {"candidate_index": 1, "mdice": 0.7950, "object_dice": 0.90},
        {"candidate_index": 2, "mdice": 0.7949, "object_dice": 1.00},
    ]
    assert select_structure_aware_candidate(candidates)["candidate_index"] == 1


def test_early_stopping_reselects_from_complete_validation_history() -> None:
    stopper = EarlyStopping(patience=20)
    assert stopper.update(0.800, 0.90)
    assert not stopper.update(0.803, 0.80)
    assert stopper.update(0.806, 0.10)
    assert stopper.selected_index == 1
    assert stopper.best == 0.803
    assert stopper.best_secondary == 0.80


def test_lightweight_candidate_state_restores_nonlatest_winner() -> None:
    class TinyHPState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(2, 2)
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.cluster1 = torch.nn.Linear(2, 2)
            self.cluster2 = torch.nn.Linear(2, 2)
            self.ema_model1 = torch.nn.Linear(2, 2)
            self.ema_model2 = torch.nn.Linear(2, 2)

    model = TinyHPState()
    project_head = torch.nn.Linear(2, 2)
    cluster_probe = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        list(model.cluster1.parameters())
        + list(model.cluster2.parameters())
        + list(project_head.parameters())
    )
    cluster_optimizer = torch.optim.Adam(cluster_probe.parameters())
    stopper = EarlyStopping(patience=20)
    snapshots: list[dict[str, object]] = []

    for value, (mdice, object_dice) in enumerate(
        [(0.800, 0.90), (0.803, 0.80), (0.806, 0.10)], start=1
    ):
        with torch.no_grad():
            for module in (model, project_head, cluster_probe):
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        parameter.fill_(float(value))
        snapshots.append(
            _capture_validation_candidate_state(
                model, project_head, cluster_probe, optimizer, cluster_optimizer
            )
        )
        stopper.update(mdice, object_dice)

    assert stopper.candidates is not None
    retained = _retained_validation_candidate_indices(
        stopper.candidates, stopper.primary_tolerance
    )
    assert stopper.selected_index == 1
    assert stopper.selected_index in retained
    _restore_validation_candidate_state(
        snapshots[stopper.selected_index],
        model,
        project_head,
        cluster_probe,
        optimizer,
        cluster_optimizer,
    )
    torch.testing.assert_close(
        model.cluster1.weight, torch.full_like(model.cluster1.weight, 2.0)
    )
    torch.testing.assert_close(
        project_head.weight, torch.full_like(project_head.weight, 2.0)
    )
    torch.testing.assert_close(
        cluster_probe.weight, torch.full_like(cluster_probe.weight, 2.0)
    )
