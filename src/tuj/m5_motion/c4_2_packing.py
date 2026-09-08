"""Backward-compatible import for the former C4-2-only packing policy."""

from tuj.m5_motion.packing import PackingKeyframeProvider


class C4_2PackingKeyframeProvider(PackingKeyframeProvider):
    """Compatibility alias; behavior is now environment-agnostic."""


__all__ = ["C4_2PackingKeyframeProvider"]
