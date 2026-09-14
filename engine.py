"""Compatibility exports for the canonical 2023HP training/evaluation loop."""

from __future__ import annotations

try:
    from .pipeline import (  # type: ignore
        _checkpoint_payload,
        _mapping_scores,
        calculate_metrics,
        evaluate_validation_mapping,
        test,
        train,
    )
except ImportError:
    from pipeline import (  # type: ignore
        _checkpoint_payload,
        _mapping_scores,
        calculate_metrics,
        evaluate_validation_mapping,
        test,
        train,
    )


__all__ = [
    "_checkpoint_payload",
    "_mapping_scores",
    "calculate_metrics",
    "evaluate_validation_mapping",
    "test",
    "train",
]

