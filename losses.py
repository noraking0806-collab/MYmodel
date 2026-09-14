"""Compatibility exports for the retained HP training objectives."""

from __future__ import annotations

try:
    from .pipeline import build_reference_pools, hidden_positive_loss  # type: ignore
    from .sparse_region_graph import SparseRegionGraphConsistency  # type: ignore
except ImportError:
    from pipeline import build_reference_pools, hidden_positive_loss  # type: ignore
    from sparse_region_graph import SparseRegionGraphConsistency


__all__ = [
    "SparseRegionGraphConsistency",
    "build_reference_pools",
    "hidden_positive_loss",
]
