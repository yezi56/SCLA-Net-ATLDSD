# FasterNet Weight Cache

This directory is reserved for official FasterNet pretrained checkpoints.
Checkpoint binaries are ignored by Git.

## Target Files

| Variant | Expected local file | Official URL | SHA256 |
|---|---|---|---|
| FasterNet-T1 | `fasternet_t1-epoch.291-val_acc1.76.2180.pth` | `https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t1-epoch.291-val_acc1.76.2180.pth` | `5D85FB083ECE3B63337D90959034A78DBBD7B0C434D73D05FAB09BF880C4646C` |
| FasterNet-T2 | `fasternet_t2-epoch.289-val_acc1.78.8860.pth` | `https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t2-epoch.289-val_acc1.78.8860.pth` | `8424CB882A4797BB73E6B7465223815894DC96876DD2EFE15E89B0B7AE785B73` |

Both files were loaded with `torch.load(..., map_location="cpu")` successfully.
