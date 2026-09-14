"""Compatibility exports for the canonical 2023HP data layer.

The implementation lives in :mod:`pipeline`, which is a direct adaptation of
``2023HP-main/hp_gland_pipeline.py``.  Keeping these names available avoids
breaking callers that used the earlier modular MYmodel layout.
"""

from __future__ import annotations

try:  # Package import (``import MYmodel.data``)
    from .pipeline import (  # type: ignore
        GlandFiles,
        HPTrainDataset,
        _evaluation_tensor,
        _read_names,
        _resolve_png,
    )
except ImportError:  # Script-style import from inside MYmodel
    from pipeline import (  # type: ignore
        GlandFiles,
        HPTrainDataset,
        _evaluation_tensor,
        _read_names,
        _resolve_png,
    )


__all__ = [
    "GlandFiles",
    "HPTrainDataset",
    "_evaluation_tensor",
    "_read_names",
    "_resolve_png",
]

