"""Compatibility exports for the retained HP component pipelines."""

from __future__ import annotations

try:
    from .multi_receptive_field import MultiReceptiveFieldAdapter  # type: ignore
    from .pipeline import (  # type: ignore
        HPFeaturizer,
        _cluster_logits,
        _load_torch_file,
        _strip_pretrained_prefixes,
        model_name_for_variant,
    )
except ImportError:
    from multi_receptive_field import MultiReceptiveFieldAdapter
    from pipeline import (  # type: ignore
        HPFeaturizer,
        _cluster_logits,
        _load_torch_file,
        _strip_pretrained_prefixes,
        model_name_for_variant,
    )


__all__ = [
    "HPFeaturizer",
    "MultiReceptiveFieldAdapter",
    "_cluster_logits",
    "_load_torch_file",
    "_strip_pretrained_prefixes",
    "model_name_for_variant",
]
