from __future__ import annotations

import torch.nn as nn

from .attention import (
    BAMBlock,
    CAA,
    CBAMBlock,
    CoordAttention,
    CPCA,
    CPAMAttention,
    CrissCrossAttention,
    DSAMAttention,
    DoubleAttention,
    ELAAttention,
    EMCAM,
    EMAAttention,
    EfficientChannelAttention,
    FCHiLoAttention,
    GAMAttention,
    GCTAttention,
    GhostModule,
    GlobalContextBlock,
    LSKAttention,
    MLCAAttention,
    PyramidPoolingPlugin,
    RCMAttention,
    SCSAAttention,
    SEAttention,
    SHSAAttention,
    ShuffleAttention,
    SimAM,
    SPGCTAttention,
    SKAttention,
    ScSEAttention,
    StripPoolingAttention,
    TripletAttention,
)


AVAILABLE_ATTENTIONS = {
    "a2": DoubleAttention,
    "double": DoubleAttention,
    "double_attention": DoubleAttention,
    "bam": BAMBlock,
    "cbam": CBAMBlock,
    "caa": CAA,
    "ca": CoordAttention,
    "coord": CoordAttention,
    "coordatt": CoordAttention,
    "coord_attention": CoordAttention,
    "cpca": CPCA,
    "cpam": CPAMAttention,
    "cc": CrissCrossAttention,
    "cca": CrissCrossAttention,
    "criss_cross": CrissCrossAttention,
    "criss_cross_attention": CrissCrossAttention,
    "dsam": DSAMAttention,
    "eca": EfficientChannelAttention,
    "ela": ELAAttention,
    "emcam": EMCAM,
    "ema": EMAAttention,
    "fchilo": FCHiLoAttention,
    "hi_lo": FCHiLoAttention,
    "gam": GAMAttention,
    "gct": GCTAttention,
    "gc": GlobalContextBlock,
    "gcnet": GlobalContextBlock,
    "global_context": GlobalContextBlock,
    "ghost": GhostModule,
    "lsk": LSKAttention,
    "mlca": MLCAAttention,
    "ppm": PyramidPoolingPlugin,
    "pyramid": PyramidPoolingPlugin,
    "pyramid_pooling": PyramidPoolingPlugin,
    "rcm": RCMAttention,
    "rectangular_self_calibration": RCMAttention,
    "sa": ShuffleAttention,
    "scsa": SCSAAttention,
    "scsa_attention": SCSAAttention,
    "scse": ScSEAttention,
    "se": SEAttention,
    "shsa": SHSAAttention,
    "shuffle": ShuffleAttention,
    "simam": SimAM,
    "sk": SKAttention,
    "sknet": SKAttention,
    "sp": StripPoolingAttention,
    "sp_gct": SPGCTAttention,
    "strip_gct": SPGCTAttention,
    "strip_pooling_gct": SPGCTAttention,
    "strip": StripPoolingAttention,
    "strip_pooling": StripPoolingAttention,
    "strip_pooling_attention": StripPoolingAttention,
    "ta": TripletAttention,
    "triplet": TripletAttention,
}


def build_attention(attention_type: str | None, channels: int, **kwargs) -> nn.Module:
    if not attention_type:
        return nn.Identity()
    attention_type = attention_type.lower().strip()
    if attention_type in {"none", "identity"}:
        return nn.Identity()
    if attention_type not in AVAILABLE_ATTENTIONS:
        raise ValueError(
            f"Unsupported attention `{attention_type}`. Available: {', '.join(sorted(AVAILABLE_ATTENTIONS))}"
        )
    return AVAILABLE_ATTENTIONS[attention_type](channels, **kwargs)
