"""Disease severity estimation from semantic masks."""

from .mask_ratio import SeverityResult, estimate_severity

__all__ = ["SeverityResult", "estimate_severity"]
