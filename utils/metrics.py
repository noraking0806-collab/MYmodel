"""Compatibility exports for the canonical 2023HP test metrics."""

from __future__ import annotations

try:
    from ..pipeline import (  # type: ignore
        _mean_metrics,
        _overlay,
        _summary,
        calculate_metrics,
        save_visual,
    )
except ImportError:
    from pipeline import (  # type: ignore
        _mean_metrics,
        _overlay,
        _summary,
        calculate_metrics,
        save_visual,
    )


def require_cv2():
    """Retain the historical helper while using the pipeline's OpenCV path."""
    import cv2

    return cv2


__all__ = [
    "calculate_metrics",
    "save_visual",
    "require_cv2",
    "_mean_metrics",
    "_overlay",
    "_summary",
]

