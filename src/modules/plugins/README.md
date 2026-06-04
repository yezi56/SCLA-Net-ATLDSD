# Plugins

This directory contains stable hot-pluggable modules used by model code.

- `factory.py` exposes `build_attention(attention_type, channels, **kwargs)` for
  NCHW feature-map plug-ins.
- `attention/` provides named files such as `cbam.py`, `eca.py`, and `ema.py`
  so each plug-in can be found directly.
- `modules.py` contains attention, context, and feature-enhancement blocks that
  keep input/output shape as `[B, C, H, W]`.
- `losses.py` exposes `build_loss(loss_type, **kwargs)` for reusable loss
  functions.
- `injector.py` contains helper utilities for attaching feature plug-ins to
  existing models.

Available `attention_type` names include:

```text
a2, bam, ca, caa, cbam, cc, cpam, cpca, criss_cross, dsam, eca, ela, ema,
emcam, gam, gc, ghost, lsk, mlca, ppm, sa, scsa, scse, se, shsa, shuffle,
simam, sk, sp, strip_pooling, ta, triplet
```

Available `loss_type` names include:

```text
ce, cross_entropy, focal, focal_loss
```
