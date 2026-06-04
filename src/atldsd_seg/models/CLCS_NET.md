# CLCS-Net Implementation Notes

CLCS-Net means **Compositional Leaf-lesion Component-aware Segmentation
Network**. The current implementation starts with the compositional part.

## Why It Is Single-stage

The model receives one RGB image and runs one forward pass. It does not run a
leaf model first and a lesion model second.

```text
image -> shared DeepLabV3+ encoder/decoder -> three output heads
```

The three heads are:

```text
leaf head:    background vs leaf
lesion head:  non-lesion vs lesion
disease head: rust / alternaria / gray / brown
```

They are composed into the final ATLDSD 6-class prediction:

```text
background   = not leaf
healthy leaf = leaf and not lesion
disease      = leaf and lesion and disease type
```

## Labels Generated From ATLDSD Masks

No extra annotation is required. The original mask is enough.

Original ATLDSD labels:

```text
0 background
1 leaf
2 rust
3 alternaria_leaf_spot
4 gray_spot
5 brown_spot
```

Derived labels:

```text
leaf target:
  0 -> 0
  1/2/3/4/5 -> 1

lesion target:
  0/1 -> 0
  2/3/4/5 -> 1

disease target:
  2 -> 0 rust
  3 -> 1 alternaria_leaf_spot
  4 -> 2 gray_spot
  5 -> 3 brown_spot
  0/1 -> ignore
```

## Code Files

```text
src/atldsd_seg/models/clcs_deeplabv3plus.py
src/atldsd_seg/losses/compositional.py
```

## Current Loss

```text
loss =
  final CE
  + 0.4 * leaf CE
  + 0.8 * lesion CE
  + 0.6 * disease CE
```

The final CE is computed on the composed 6-class output. The other three losses
make the three heads learn the biological structure directly.

## Paper Claim

Compared with DUNet-style two-stage segmentation:

```text
DUNet: DeepLabV3+ leaf segmentation -> U-Net lesion segmentation
CLCS-Net: one model, one inference, structured leaf/lesion/disease heads
```

The paper claim should be:

```text
CLCS-Net encodes the leaf-lesion-disease hierarchy of ATLDSD into the output
space and directly supports severity estimation through lesion / leaf.
```
