# DeepLabV3+ Integration Source Notes

## RepConv decoder block

- GitHub reference checked on 2026-06-01:
  - `ultralytics/ultralytics`: https://github.com/ultralytics/ultralytics
    - Stars from GitHub API: 57,841
    - License: AGPL-3.0
    - Relevant public API: `ultralytics.nn.modules.conv.RepConv`
  - `DingXiaoH/RepVGG`: https://github.com/DingXiaoH/RepVGG
    - Stars from GitHub API: 3,472
    - License: MIT
    - Relevant method: structural re-parameterization with 3x3, 1x1, and identity BN branches

The local `RepConvBlock` in `nets/deeplabv3_plus.py` is a compact decoder-focused implementation of the RepConv/RepVGG idea. It keeps the DeepLabV3+ decoder API stable through `--decoder-conv-type repconv` and avoids adding YOLO-specific dependencies.
