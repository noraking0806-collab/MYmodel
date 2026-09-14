# MYmodel: retained HP pipelines

This directory contains the HP baseline, the retained MRFA/SGC ablations, and
the selected final model. All unrelated experimental branches have been
removed so that the implementation, wrappers, tests, and saved outputs expose
the same eight pipelines.

## Retained pipelines

| Pipeline | Variant | Purpose |
|---|---|---|
| `2023HP` | `none` | Source-aligned HP baseline |
| `HP-SGC` | `sgc` | HP plus sparse region-graph consistency |
| `HP-MRFA12` | `mrfa12` | HP plus the dilation-{1,2} MRFA adapter |
| `HP-MRFA1-SGC` | `mrfa1_sgc` | Dilation-1 MRFA branch ablation plus SGC |
| `HP-MRFA2-SGC` | `mrfa2_sgc` | Dilation-2 MRFA branch ablation plus SGC |
| `HP-MRFA12-SGC` | `mrfa12_sgc` | Full dilation-{1,2} MRFA plus SGC; final model |
| `HP-MRFA12-SGC-ATTR` | `mrfa12_sgc_attr` | Attraction-only SGC loss ablation |
| `HP-MRFA12-SGC-REP` | `mrfa12_sgc_rep` | Repulsion-only SGC loss ablation |

The `best` alias resolves to `HP-MRFA12-SGC`.

## Method

`2023HP` retains the released HP design: frozen DINO-S/8 features,
task-agnostic and task-specific hidden positives, a momentum teacher head,
local hidden-positive mixing, and an unsupervised two-cluster probe. Training
masks are never opened. Validation labels are used only for cluster-permutation
selection and structure-aware early stopping; test labels are used only for
reporting.

MRFA is a trainable residual multi-receptive-field adapter placed on the DINO
feature grid. The retained full adapter uses dilations `{1,2}`. The `MRFA1`
and `MRFA2` variants change only this branch set to `{1}` or `{2}`.

SGC is parameter-free. It pools online and EMA codes to a sparse region graph,
uses directed Top-K semantic neighbours for attraction, and applies repulsion
to low-similarity spatial boundaries. Its loss is

```text
L_SGC = w_attr * L_attraction + w_rep * L_repulsion
```

The full model uses `w_attr=1.0` and `w_rep=0.5`. `ATTR` sets only
`w_rep=0`; `REP` sets only `w_attr=0`. The overall training loss is

```text
L = L_HP + alpha * L_consistency + L_cluster
    + graph_weight * lambda_HP * L_SGC
```

## Formal GLaS results

All entries below use the same fixed split, seed 42, original-resolution test
metrics, and 80 test images.

| Pipeline | Mean IoU | Gland Dice | Mean Dice | Object Dice |
|---|---:|---:|---:|---:|
| `2023HP` | 68.58% | 81.58% | 80.46% | 63.48% |
| `HP-SGC` | 71.19% | 82.80% | 82.56% | 67.56% |
| `HP-MRFA12` | 67.11% | 80.80% | 79.40% | 61.05% |
| `HP-MRFA1-SGC` | 71.64% | 83.27% | 82.89% | 67.07% |
| `HP-MRFA2-SGC` | 71.70% | 83.46% | 82.94% | 66.56% |
| **`HP-MRFA12-SGC` (final)** | **72.35%** | **83.67%** | **83.37%** | 67.55% |
| `HP-MRFA12-SGC-ATTR` | 67.51% | 81.04% | 79.76% | 60.39% |
| `HP-MRFA12-SGC-REP` | 71.89% | 83.22% | 83.05% | 66.36% |

`HP-MRFA12-SGC` is retained as the final model because it gives the strongest
Mean IoU, Gland Dice, and Mean Dice among the retained experiments. `HP-SGC`
has a 0.01 percentage-point Object-Dice advantage, while the full model is
materially better on the three primary pixel metrics.

## Running

Place the pretrained DINO checkpoint at
`dino_deitsmall8_300ep_pretrain.pth` before training or evaluation. Model
weights, checkpoints, and bulk prediction images are intentionally kept out of
Git; compact experiment metrics and summaries remain versioned.

Train or test one pipeline:

```powershell
python train.py --pipeline HP-MRFA12-SGC --dataset glas
python test.py --pipeline HP-MRFA12-SGC --dataset glas
```

Use the selected-model alias:

```powershell
python train.py --pipeline best --dataset glas
python test.py --pipeline best --dataset glas
```

Run all eight retained pipelines on all three datasets:

```powershell
python train.py --pipeline all --dataset all
python test.py --pipeline all --dataset all
```

Accepted short aliases include `hp`, `sgc`, `mrfa12`, `mrfa1-sgc`,
`mrfa2-sgc`, `mrfa12-sgc`, `mrfa12-sgc-attr`, `mrfa12-sgc-rep`, and their
underscore forms.

## Retained layout

| Path | Role |
|---|---|
| `pipeline.py` | HP/MRFA/SGC implementation, training, checkpointing, evaluation |
| `train.py` | Eight-pipeline training registry and launcher |
| `test.py` | Eight-pipeline checkpoint evaluation registry and launcher |
| `multi_receptive_field.py` | MRFA adapter |
| `sparse_region_graph.py` | Directed sparse graph loss |
| `model/`, `utils/` | HP backbone and shared utilities |
| `tests/test_hp_sgc.py` | Retained registry, ablation, gradient, and restore tests |
| `tests/smoke_core_ablations.py` | Four new ablation train/reload/test smoke run |
| `train_out/<pipeline>/` | Compact training records; checkpoints stay local |
| `test_out/<pipeline>/` | Metrics and summaries; prediction images stay local |
| `2023HP.pdf`, `2023HP_extracted.txt` | Local-only HP source/provenance material |

The selected GLaS scores are summarized above. Detailed ablation interpretation
is in `OPTIMIZATION_RESULTS.md`.
