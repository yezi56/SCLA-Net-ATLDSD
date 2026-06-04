"""Stable pluggable modules for model experiments."""

from .factory import AVAILABLE_ATTENTIONS, build_attention
from .injector import AttentionHookSpec, attach_attention_hooks
from .losses import AVAILABLE_LOSSES, build_loss

__all__ = [
    "AVAILABLE_ATTENTIONS",
    "AVAILABLE_LOSSES",
    "AttentionHookSpec",
    "attach_attention_hooks",
    "build_attention",
    "build_loss",
]
