# Model

Place trained checkpoints, exported deployment weights, and released model artifacts here.

Large binary artifacts are intentionally ignored by git:

```text
*.pth
*.pt
*.onnx
checkpoints/
```

Recommended release files:

```text
model/best_miou_weights.pth
model/deploy_repconv_fused.pth
model/model_card.md
```

RepConv helper scripts:

```bash
python docx_pipeline/scripts/check_deeplab_repconv_deploy.py --help
python docx_pipeline/scripts/export_deeplab_repconv_deploy.py --help
```
