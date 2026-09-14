"""Command-line compatibility entry point for the retained HP pipelines."""

from __future__ import annotations

try:
    from .pipeline import build_parser, main  # type: ignore
except ImportError:
    from pipeline import build_parser, main  # type: ignore


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
