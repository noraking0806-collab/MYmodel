# Retained HP ablation results

## Decision

The final model is `HP-MRFA12-SGC`. The retained experiment family contains
the `2023HP` baseline, two standalone component controls, two MRFA branch
ablations, the complete MRFA12+SGC model, and two SGC loss-component
ablations. All results are from the fixed GLaS comparison split at seed 42.

## Complete retained result table

| Pipeline | Mean IoU | Gland Dice | Mean Dice | Object Dice |
|---|---:|---:|---:|---:|
| `2023HP` | 68.58% | 81.58% | 80.46% | 63.48% |
| `HP-SGC` | 71.19% | 82.80% | 82.56% | 67.56% |
| `HP-MRFA12` | 67.11% | 80.80% | 79.40% | 61.05% |
| `HP-MRFA1-SGC` | 71.64% | 83.27% | 82.89% | 67.07% |
| `HP-MRFA2-SGC` | 71.70% | 83.46% | 82.94% | 66.56% |
| **`HP-MRFA12-SGC`** | **72.35%** | **83.67%** | **83.37%** | 67.55% |
| `HP-MRFA12-SGC-ATTR` | 67.51% | 81.04% | 79.76% | 60.39% |
| `HP-MRFA12-SGC-REP` | 71.89% | 83.22% | 83.05% | 66.36% |

## Component attribution

Relative to `2023HP`, SGC alone improves Mean IoU by 2.61 percentage points,
Mean Dice by 2.10 points, and Object Dice by 4.08 points. MRFA12 alone does not
improve this run, but MRFA12 combined with SGC reaches the best retained pixel
metrics. This interaction supports treating MRFA as a feature adaptation for
the graph objective, not as a standalone replacement for HP.

## MRFA branch ablation

| MRFA branches with SGC | Mean IoU | Gland Dice | Mean Dice | Object Dice |
|---|---:|---:|---:|---:|
| dilation `{1}` | 71.64% | 83.27% | 82.89% | 67.07% |
| dilation `{2}` | 71.70% | 83.46% | 82.94% | 66.56% |
| dilation `{1,2}` | **72.35%** | **83.67%** | **83.37%** | **67.55%** |

Neither single branch reproduces the full result. The two-radius adapter gains
0.65-0.71 points in Mean IoU and 0.40-0.48 points in Mean Dice over the
single-branch variants, supporting complementary local and wider context.

## SGC loss decomposition

| SGC terms with MRFA12 | Mean IoU | Gland Dice | Mean Dice | Object Dice |
|---|---:|---:|---:|---:|
| attraction only | 67.51% | 81.04% | 79.76% | 60.39% |
| repulsion only | 71.89% | 83.22% | 83.05% | 66.36% |
| attraction + repulsion | **72.35%** | **83.67%** | **83.37%** | **67.55%** |

Repulsion contributes most of SGC's gain in this run. Attraction alone is not
sufficient, while adding attraction to repulsion still improves all four
metrics: +0.46 Mean IoU, +0.45 Gland Dice, +0.32 Mean Dice, and +1.19 Object
Dice. The retained full loss is therefore justified by the decomposition.

## Final-model artifact

- Pipeline: `HP-MRFA12-SGC`
- Variant: `mrfa12_sgc`
- Selected validation epoch: 73
- Checkpoint: `train_out/HP-MRFA12-SGC/glas/checkpoint_final.pth`
- Validation Mean Dice: 86.1724%
- Validation Gland Dice: 85.5336%
- Validation Object Dice: 71.0841%
- Test samples: 80 at original resolution

The `best` alias in both wrappers resolves to this pipeline.

## Limits on interpretation

These are single-seed results with a nine-image validation split. They support
component attribution within this controlled experiment, but they do not
estimate training-seed variance or establish significance across cohorts.
