# Data

This folder contains dataset preparation utilities and layout notes.

The ATLDSD dataset itself is not included in this repository. Use a VOC-style layout:

```text
VOCdevkit/
  VOC2012/
    JPEGImages/
    SegmentationClass/
    ImageSets/Segmentation/
  VOC2007/
```

Recommended local path:

```text
D:\dataset\ATLDSD\VOCdevkit
```

You may also set:

```bash
export ATLDSD_VOCDEVKIT_PATH=/absolute/path/to/VOCdevkit
```

Utilities:

```text
data/tools/prepare_atldsd_voc.py
```
